import abc

import basix
import dolfinx as dfx
from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm, SUM

import numpy as np

from petsc4py import PETSc

import pyvista as pv

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    create_1p1_DFN_mesh,
    create_particle_summation_measure,
    compute_chemical_potential,
    _free_energy as free_energy)

from pyMoBiMP.fenicsx_utils import (
    NewtonSolver,
    RuntimeAnalysisBase,
    StopEvent,
    strip_off_xdmf_file_ending)


# %% Helper functions
# ===================

def log(*msg, my_rank=0, all_procs=False):

    rank = MPI.COMM_WORLD.rank

    if not all_procs:
        if rank == my_rank:
            print("[0] ", *msg, flush=True)

    else:
        size = MPI.COMM_WORLD.size
        digits = np.ceil(np.log10(size))

        print(f"[{rank:0{digits}}] ", *msg)


def time_stepping(
    solver,
    u,
    u0,
    T,
    dt,
    V_cell,
    t_start=0,
    dt_max=10.0,
    dt_min=1e-9,
    dt_increase=1.1,
    tol=1e-6,
    event_handler=lambda t, **pars: None,
    output=None,
    runtime_analysis=None,
    logging=True,
    callback=lambda: None,
    **event_pars,
):

    assert dt_min < dt_max
    assert tol > 0.

    t = t_start
    dt.value = dt_min * dt_increase

    # Make sure initial time step does not exceed limits.
    dt.value = np.minimum(dt.value, dt_max)

    # Prepare output
    if output is not None:
        output = np.atleast_1d(output)
        [o.save_snapshot(u, t, force=True) for o in output]

    if runtime_analysis is not None:
        runtime_analysis.analyze(t)

    it = 0

    while t < T:

        it += 1

        try:
            u.x.scatter_forward()
            u0.x.array[:] = u.x.array[:]
            u0.x.scatter_forward()

            if float(dt) < dt_min:
                raise ValueError(f"Timestep too small (dt={dt.value})!")

            t += float(dt)

            V_cell.update()
            callback()  # check current
            stop = event_handler(t, cell_voltage=V_cell.value, **event_pars)

            if stop:
                break

            iterations, success = solver.solve(u)

            if not success:
                raise RuntimeError("Newton solver did not converge.")

        except StopEvent as e:

            print(e)
            print(">>> Stop integration.")

            break

        except (RuntimeError, AssertionError) as e:

            print(e)

            # reset and continue with smaller time step.
            u.x.array[:] = u0.x.array[:]

            iterations = solver.max_it

            if dt.value > dt_min:

                # Reset the timestep
                t -= float(dt)
                it -= 1

                # Lower the timestep size ...
                dt.value *= 0.5

                # ... reset the solution array ...
                u.x.array[:] = u0.x.array[:]
                u.x.scatter_forward()

                print(f"Decrease timestep to dt={dt.value:1.3e}")

                # ... and restart the current iteration.
                continue

            else:
                if output is not None:

                    [o.save_snapshot(u, t) for o in output]

        except ValueError as e:

            print(e)

            if output is not None:
                [o.save_snapshot(u, t, force=True) for o in output]

            break

        # Adaptive timestepping a la Yibao Li et al. (2017)
        u_max_loc = np.abs(u.sub(0).x.array - u0.sub(0).x.array).max()

        u_err_max = u.function_space.mesh.comm.allreduce(u_max_loc, op=MPI.MAX)

        if iterations < solver.max_it / 5:
            # Use the given increment factor if we are in a safe region, i.e.,
            # if the Newton solver converges sufficiently fast.
            inc_factor = dt_increase
        elif iterations < solver.max_it / 2:
            # Reduce the increment if we take more iterations.
            inc_factor = 1 + 0.1 * (dt_increase - 1.)
        elif iterations > solver.max_it * 0.8:
            # Reduce the timestep in case we are approaching max_it
            inc_factor = 0.9
        else:
            # Do not increase timestep between [0.5*max_it, 0.8*max_it]
            inc_factor = 1.0

        dt.value = min(max(tol / u_err_max, dt_min), dt_max, inc_factor * dt.value)

        # Find the minimum timestep among all processes.
        # Note that we explicitly use COMM_WORLD since the mesh communicator
        # only groups the processes belonging to one particle.
        dt_global = MPI.COMM_WORLD.allreduce(dt.value, op=MPI.MIN)

        dt.value = dt_global

        if runtime_analysis is not None:
            runtime_analysis.analyze(t)

        if output is not None:
            [o.save_snapshot(u, t) for o in output]

        if logging:
            perc = (t - t_start) / (T - t_start) * 100

            if MPI.COMM_WORLD.rank == 0:
                print(
                    f"{perc:>3.0f} % :",
                    f"t[{it:06}] = {t:1.6f}, "
                    f"dt = {dt.value:1.3e}, "
                    f"its = {iterations}",
                    flush=True
                )

    else:

        if output is not None:

            [o.finalize() for o in output]

    return


class NonlinearProblem(NonlinearProblemBase):
    def __init__(self, *args, callback=lambda: None, **kwargs):
        super().__init__(*args, **kwargs)

        self.callback = callback

    def form(self, x):
        super().form(x)

        self.callback()


def plot_solution_on_grid(u):

    V = u.function_space

    topology, cell_types, x = dfx.plot.vtk_mesh(V)

    n_particles = np.max(x)
    x[:, 1] /= n_particles

    grid = pv.UnstructuredGrid(topology, cell_types, x)

    grid['u'] = u.x.array

    plotter = pv.Plotter()

    warped = grid.warp_by_scalar('u')

    plotter.add_mesh(warped, show_edges=True, show_vertices=False, show_scalar_bar=True)
    plotter.add_axes()
    plotter.add_bounding_box()

    plotter.show()


class DefaultPhysicalSetup:

    L_mean: float = 10.
    L_var_rel: float = 0.1

    def __init__(self, V):
        self.function_space = V

        self.mesh = V.mesh

        self.dA = create_particle_summation_measure(self.mesh)

        # auxiliary space for coefficient functions
        self._V0, _ = V.sub(0).collapse()

        self._setup_particle_radii()

        self._setup_particle_surfaces()
        self._setup_total_surface()
        self._setup_surface_weights()

        self._setup_reaction_affinity()
        self._setup_mean_affinity()

    def _setup_particle_radii(self):
        # particle parameters
        Rs = dfx.fem.Function(self._V0)
        Rs.x.array[:] = 1.
        Rs.x.scatter_forward()

        self._Rs = Rs

    @property
    def particle_radii(self):
        return self._Rs

    def _setup_particle_surfaces(self):

        self._As = 4 * np.pi * self._Rs**2

    @property
    def particle_surfaces(self):
        return self._As

    def _setup_total_surface(self):
        A_ufl = self._As * self.dA

        A = dfx.fem.assemble_scalar(dfx.fem.form(A_ufl))
        A = self.mesh.comm.allreduce(A, op=SUM)

        self._A = A

    @property
    def total_surface(self):
        return self._A

    def _setup_surface_weights(self):
        self._a_ratios = self._As / self._A

    @property
    def surface_weights(self):
        return self._a_ratios

    def _setup_reaction_affinity(self):

        Ls = dfx.fem.Function(self._V0)
        Ls.interpolate(
            lambda x: self.L_mean * (1. + self.L_var_rel * (2 * x[1] - 1.)))

        self._Ls = Ls

    @property
    def reaction_affinities(self):
        return self._Ls

    def _setup_mean_affinity(self):
        # Weighted mean reaction affinity parameter taken
        # from particle surfaces.
        L_ufl = self._a_ratios * self._Ls * self.dA

        L = dfx.fem.assemble_scalar(dfx.fem.form(L_ufl))
        L = self.mesh.comm.allreduce(L, op=SUM)

        self._L = L

    @property
    def mean_affinity(self):
        return self._L

    def total_surface_and_weights(self):
        return self.total_surface, self.surface_weights

    def mean_and_particle_affinities(self):
        return self.mean_affinity, self.reaction_affinities


class Voltage(dfx.fem.Constant):

    def __init__(self,
                 u: dfx.fem.Function,
                 I_global: float | dfx.fem.Constant,
                 physical_setup: DefaultPhysicalSetup):

        self.u = u
        self.function_space = u.function_space
        self.I_global = I_global

        self.physical_setup = physical_setup

        A, a_ratios = self.physical_setup.total_surface_and_weights()
        L, Ls = self.physical_setup.mean_and_particle_affinities()

        dA = self.physical_setup.dA

        _, mu = ufl.split(u)

        V_cell_ufl = - mu * Ls / L * a_ratios * dA
        V_cell_ufl -= I_global / L * a_ratios * dA

        self.V_cell_cpp = dfx.fem.form(V_cell_ufl)

        super().__init__(self.function_space.mesh,
                         self.compute_voltage())

    def compute_voltage(self):

        V_cell_value = float(dfx.fem.assemble_scalar(self.V_cell_cpp))
        voltage = comm.allreduce(V_cell_value, op=SUM)

        return voltage

    def update(self):
        voltage = self.compute_voltage()

        np.copyto(self._cpp_object.value, np.asarray(voltage))

    @property
    def value(self):

        self.update()

        return float(self._cpp_object.value)

    @property
    def form(self):
        return self.V_cell_cpp


class TestCurrent():
    def __init__(self, u, V_cell):

        _, mu = ufl.split(u)

        a_ratios = V_cell.physical_setup.surface_weights
        Ls = V_cell.physical_setup.reaction_affinities

        dA = create_particle_summation_measure(u.function_space.mesh)

        I_particle = - Ls * (mu + V_cell)

        I_global_ref_ufl = a_ratios * I_particle * dA

        self.I_global_ref_form = dfx.fem.form(I_global_ref_ufl)
        self.I_global = V_cell.I_global
        self.V_cell = V_cell

    def compute_current(self):

        I_global_ref = dfx.fem.assemble_scalar(self.I_global_ref_form)
        I_global_ref = comm.allreduce(I_global_ref, op=SUM)

        return I_global_ref

    def __call__(self):

        I_global_ref = self.compute_current()

        if not np.isclose(I_global_ref, self.I_global.value):
            raise AssertionError(
                "Error in global current computation" +
                f"I_global_ref = {I_global_ref} " +
                f"!= {self.I_global.value} = I_global.value")

        return I_global_ref


class AnalyzeOCP(RuntimeAnalysisBase):
    def setup(self, u_state, c_of_y, V_cell, *args, **kwargs):
        super().setup(u_state, *args, **kwargs)

        # Function space(s) and mesh information
        V = self.u_state.function_space
        mesh = V.mesh

        V0, _ = V.sub(0).collapse()

        coords = ufl.SpatialCoordinate(mesh)
        r = coords[0]

        dA_R = create_particle_summation_measure(mesh)

        # By integrating over the boundary we get a measure of the
        # number of particles.
        num_particles = dfx.fem.assemble_scalar(
            dfx.fem.form(dfx.fem.Constant(mesh, 1.) * dA_R)
        )
        num_particles = mesh.comm.allreduce(num_particles, op=SUM)

        y, mu = self.u_state.split()

        # compute state of charge
        c = c_of_y(y)

        self.soc_form = dfx.fem.form(3 / num_particles * c * r**2 * ufl.dx)

        self.V_cell_form = V_cell.form

    def analyze(self, t):

        mesh = self.u_state.function_space.mesh

        soc = dfx.fem.assemble_scalar(self.soc_form)
        soc = mesh.comm.allreduce(soc, op=SUM)

        V_cell = dfx.fem.assemble_scalar(self.V_cell_form)
        V_cell = mesh.comm.allreduce(V_cell, op=SUM)

        self.data.append([soc, V_cell])

        return super().analyze(t)


class ChargeDischargeExperiment():

    # Global parameters
    c_rate: float = 0.01
    v_cell_bounds = [-3.7, 3.7]
    stop_at_empty = False
    stop_on_full = False
    cycling = False
    logging = False

    def __init__(
        self,
        u: dfx.fem.Function,
    ):

        self.u = u
        # FIXME: Find out where the factor of 2 is coming from!
        self._I_charge = dfx.fem.Constant(u.function_space.mesh, 2 / 3 * self.c_rate)

    @property
    def I_charge(self):
        return self._I_charge

    def __call__(self, t, cell_voltage):
        return self.experiment(t, cell_voltage)

    def experiment(self, t, cell_voltage):

        if self.logging:
            log(
                f"t={t:1.5f} ; V_cell = {cell_voltage}")

        # Whenever you may ask yourself whether this works, mind the sign!
        # cell_voltage is the voltage computed by AnalyzeCellPotential, ie,
        # it increases with chemical potential at the surface of the particles.
        # The actual cell voltage as measured is the negative of it.
        if -cell_voltage > self.v_cell_bounds[1] and self.I_charge.value > 0.0:
            log(
                ">>> Cell voltage exceeds maximum " +
                f"(V_cell = {-cell_voltage:1.3f} > {self.v_cell_bounds[1]:1.3f})."
            )

            if self.stop_on_full:
                log(">>> Cell is filled.")

                return True

            self.I_charge.value *= -1.0

            return False

        if -cell_voltage < self.v_cell_bounds[0] and self.I_charge.value < 0.0:

            if self.stop_at_empty:
                log(">>> Cell voltage exceeds minimum." +
                    f"(V_cell = {-cell_voltage:1.3f} < {self.v_cell_bounds[0]:1.3f}).")

                return True

            else:
                if self.cycling:
                    print(">>> Start charging.")
                    self.I_charge.value *= -1.0

                else:
                    print(">>> Stop charging.")
                    self.I_charge.value = 0.0

                return False

        return False


def DFN_function_space(mesh):
    # %% The DOLFINx function space
    # -----------------------------
    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    V = dfx.fem.functionspace(mesh, basix.ufl.mixed_element([elem1, elem1]))

    return V


def DFN_FEM_form(
    u, u0, v, dt, V_cell, free_energy,
    M=lambda c: c * (1 - c), gamma=0.1, grad_c_bc=lambda c: 0.0
):

    V = u.function_space
    mesh = V.mesh
    y, mu = ufl.split(u)
    y0, mu0 = ufl.split(u0)

    v_c, v_mu = ufl.split(v)

    dA = create_particle_summation_measure(mesh)
    Ls = V_cell.physical_setup.reaction_affinities

    I_particle = - Ls * (mu + V_cell)

    theta = 1.0

    c = c_of_y(y)
    c0 = c_of_y(y0)

    r, _ = ufl.SpatialCoordinate(mesh)

    s_V = 4 * np.pi * r**2
    s_A = 2 * np.pi * r**2

    dx = ufl.dx  # The volume element

    mu_chem = compute_chemical_potential(free_energy, c)
    mu_theta = theta * mu + (theta - 1.0) * mu0

    flux = M(c) * mu_theta.dx(0)

    F1 = s_V * (c - c0) * v_c * dx
    F1 += s_V * flux * v_c.dx(0) * dt * dx
    F1 -= s_A * I_particle * v_c * dt * dA

    F2 = s_V * mu * v_mu * dx
    F2 -= s_V * mu_chem * v_mu * dx
    F2 -= gamma * (s_V * c.dx(0) * v_mu.dx(0) * dx)
    F2 += grad_c_bc(c) * (s_A * v_mu * dA)

    F = F1 + F2

    return F


class DFNSimulationBase(abc.ABC):

    PhysicalSetup = NotImplemented
    RuntimeAnalysis = NotImplemented
    Output = NotImplemented
    Experiment = NotImplemented
    free_energy = staticmethod(free_energy)

    def __init__(
            self,
            comm: MPI.Intracomm = MPI.COMM_WORLD,
            n_particles: int = 1024,
            n_radius: int = 16,
            output_destination: str = "CH_4_DFN.xdmf",
            gamma: float = 0.1):

        self.comm = comm

        self.n_particles = n_particles
        self.n_radius = n_radius

        self.create_mesh()

        self.create_function_space()

        # Shorthands
        V = self.function_space
        mesh = V.mesh

        self.u = u = dfx.fem.Function(V)
        self.u0 = u0 = dfx.fem.Function(V)

        v = ufl.TestFunction(V)

        self.dt = dt = dfx.fem.Constant(mesh, 1e-8)

        # Runtime analysis and output
        # ==============================
        self.output_file_name_base = strip_off_xdmf_file_ending(
            output_destination
        )

        self.physical_setup = self.PhysicalSetup(V)

        self.experiment = self.Experiment(u)

        self.V_cell = V_cell = Voltage(u, self.experiment.I_charge, self.physical_setup)

        self.rt_analysis = self.RuntimeAnalysis(
            u, c_of_y, V_cell, filename=self.output_file_name_base + "_rt.txt")

        self.callback = TestCurrent(u, V_cell)

        # FEM Form
        # ========
        self.F = F = DFN_FEM_form(u, u0, v, dt, V_cell, self.free_energy,
                                  gamma=gamma)

        # DOLFINx problem and solver setup
        # ===================================
        self.solver_setup(comm, u, V_cell, F)

        self.initial_data()

    def initial_data(self):

        # This corresponds to roughly c = 1e-3
        self.u0.sub(0).x.array[:] = -6.90675478

        # By setting dt=0, we have the first equation to be y=y0 and
        # solve for the chemical potential, only.
        self.dt.value = 0.

        # Do some diagnostic checks to see whether the solver did ok.
        residual = dfx.fem.form(self.F)

        log(dfx.fem.petsc.assemble_vector(residual).norm())

        its, success = self.solver.solve(self.u)
        error = dfx.fem.petsc.assemble_vector(residual).norm()
        log(its, error)
        assert np.isclose(error, 0.)

    def solver_setup(self, comm, u, V_cell, F):
        problem = NonlinearProblem(F, u)
        self.solver = solver = NewtonSolver(
            comm, problem, callback=lambda solver, uh: V_cell.update())
        solver.rtol = 1e-7
        solver.max_it = 50
        solver.convergence_criterion = "incremental"
        solver.relaxation_parameter = 1.0

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "ksp"
        ksp.setFromOptions()

        self.solver

    def create_mesh(self):

        self.mesh = create_1p1_DFN_mesh(
            self.comm, self.n_radius, self.n_particles)

    def create_function_space(self):

        self.function_space = DFN_function_space(self.mesh)

    def run(self,
            t_start: float = 0.,
            t_final: float = 150.,
            n_out: int = 501,
            dt_min: float = 1e-9,
            dt_max: float = 1e-3,
            dt_increase: float = 1.1,
            tol: float = 1e-5):

        if n_out > 0:
            self.output = self.Output(
                self.u,
                np.linspace(t_start, t_final, n_out),
                filename=self.output_file_name_base + ".xdmf",
                variable_transform=c_of_y,
            )
        else:
            self.output = None

        time_stepping(
            self.solver,
            self.u,
            self.u0,
            t_final,
            self.dt,
            self.V_cell,
            t_start=t_start,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_increase=dt_increase,
            tol=tol,
            runtime_analysis=self.rt_analysis,
            output=self.output,
            event_handler=self.experiment.experiment,
            callback=self.callback
        )
