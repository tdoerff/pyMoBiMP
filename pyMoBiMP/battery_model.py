import abc

import dolfinx as dfx
import dolfinx.fem.petsc

from mpi4py import MPI

import numpy as np

import ufl

from pyMoBiMP.fenicsx_utils import (
    assemble_scalar,
    log,
    NewtonSolver,
    NonlinearProblemBlock,
    RuntimeAnalysisBase,
    StopEvent,
    strip_off_xdmf_file_ending
)

from pyMoBiMP.dfn_utils import (
    create_1p1_DFN_mesh,
    create_particle_summation_measure,
    DFN_function_space,
)


def _free_energy(
    u: dfx.fem.Function, a: float = 6.0 / 4.0, b: float = 0.2, c: float = 5.0
):

    fe = (
        u * ufl.ln(u)
        + (1 - u) * ufl.ln(1 - u)
        + a * u * (1 - u)
        + b * ufl.sin(c * np.pi * u)
    )

    return fe


# Forward and backward variable transformation.
def c_of_y(y):
    return ufl.exp(y) / (1 + ufl.exp(y))


def y_of_c(c):
    return ufl.ln(c / (1 - c))


def compute_chemical_potential(free_energy, c):

    # Differentiate the free energy function to
    # obtain the chemical potential
    c = ufl.variable(c)
    dfdc = ufl.diff(free_energy(c), c)
    mu_chem = dfdc
    return mu_chem


def time_stepping(
    solver,
    u,
    voltage,
    u0,
    T,
    dt,
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

    u_inc_form = dfx.fem.form(ufl.dot(u - u0, u - u0) * ufl.dx)

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

            stop = event_handler(t, cell_voltage=voltage.x.array[0], **event_pars)

            if stop:
                break

            iterations, _ = solver.solve([u, voltage])

            # Callback test for the total current density. That should be
            # done after the timestep. Otherwise an unsucessul previous
            # timestep might be evaluated leading to a AssertionError that
            # again restarts the timestep with smaller step size until we
            # reach dt_min.
            callback()

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
        u_inc = assemble_scalar(u_inc_form)
        u_inc = max(u_inc, 1e-9)

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

        dt.value = min(max(tol / u_inc, dt_min), dt_max, inc_factor * dt.value)

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

        A = assemble_scalar(dfx.fem.form(A_ufl))

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

        L = assemble_scalar(dfx.fem.form(L_ufl))

        self._L = L

    @property
    def mean_affinity(self):
        return self._L

    def total_surface_and_weights(self):
        return self.total_surface, self.surface_weights

    def mean_and_particle_affinities(self):
        return self.mean_affinity, self.reaction_affinities


class TestCurrent():
    def __init__(self, u, voltage, I_global, physical_setup):

        _, mu = ufl.split(u)

        a_ratios = physical_setup.surface_weights
        Ls = physical_setup.reaction_affinities

        dA = create_particle_summation_measure(u.function_space.mesh)

        I_particle = - Ls * (mu + voltage)

        I_global_ref_ufl = a_ratios * I_particle * dA

        self.I_global_ref_form = dfx.fem.form(I_global_ref_ufl)
        self.I_global = I_global

    def compute_current(self):

        I_global_ref = assemble_scalar(self.I_global_ref_form)

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
    def setup(self, u_state, voltage, c_of_y, *args, **kwargs):
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
        num_particles = assemble_scalar(dfx.fem.form(dfx.fem.Constant(mesh, 1.) * dA_R))

        y, mu = self.u_state.split()

        # compute state of charge
        c = c_of_y(y)

        self.soc_form = dfx.fem.form(3 / num_particles * c * r**2 * ufl.dx)

        self.V_cell_form = dfx.fem.form(voltage / num_particles * ufl.dx)

    def analyze(self, t):

        soc = assemble_scalar(self.soc_form)

        V_cell = assemble_scalar(self.V_cell_form)

        self.data.append([soc, V_cell])

        return super().analyze(t)


class ChargeDischargeExperiment():

    # Global parameters
    c_rate: float = 0.01
    v_cell_bounds = [-4, 4]
    stop_at_empty = False
    stop_on_full = False
    cycling = False
    logging = False  # TODO: use logger, set loglevel info

    def __init__(
        self,
        u: dfx.fem.Function,
    ):

        self.u = u
        self._I_charge = dfx.fem.Constant(u.function_space.mesh, 1 / 3 * self.c_rate)

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


class DFNSimulationBase(abc.ABC):

    @staticmethod
    def free_energy(u):
        return _free_energy(u)

    @staticmethod
    def mobility(c):
        return c * (1 - c)

    def __init__(
            self,
            comm: MPI.Intracomm = MPI.COMM_WORLD,
            n_particles: int = 1024,
            n_radius: int = 16,
            output_destination: str = "CH_4_DFN.xdmf",
            gamma: float = 0.1,
            **solver_args):

        # Ensure that the subclass has defined the required attributes
        if not hasattr(self, 'PhysicalSetup'):
            raise NotImplementedError("Subclasses must define 'PhysicalSetup'")

        if not hasattr(self, 'RuntimeAnalysis'):
            raise NotImplementedError("Subclasses must define 'RuntimeAnalysis'")

        if not hasattr(self, 'Experiment'):
            raise NotImplementedError("Subclasses must define 'Experiment'")

        if not hasattr(self, 'Output'):
            raise NotImplementedError("Subclasses must define 'Output'")

        self.comm = comm

        self.n_particles = n_particles
        self.n_radius = n_radius

        self.create_mesh()

        self.create_function_spaces()

        # Shorthands
        V, W = self.function_spaces
        mesh = V.mesh

        self.u = u = dfx.fem.Function(V)
        self.u0 = u0 = dfx.fem.Function(V)

        v = ufl.TestFunction(V)

        self.voltage = voltage = dfx.fem.Function(W)
        v_voltage = ufl.TestFunction(W)

        self.dt = dt = dfx.fem.Constant(mesh, 1e-8)

        # Runtime analysis and output
        # ==============================
        self.output_file_name_base = strip_off_xdmf_file_ending(
            output_destination
        )

        self.physical_setup = self.PhysicalSetup(V)

        self.experiment = self.Experiment(u)

        self.rt_analysis = self.RuntimeAnalysis(
            u, voltage, c_of_y, filename=self.output_file_name_base + "_rt.txt"
        )

        self.callback = TestCurrent(
            u, voltage, self.experiment.I_charge, self.physical_setup
        )

        # FEM Form
        # ========
        self.F = DFN_FEM_form(
            u,
            voltage,
            u0,
            v,
            dt,
            self.free_energy,
            self.physical_setup.reaction_affinities,
            M=lambda c: self.mobility(c),
            gamma=gamma,
        )

        self.voltage_form = voltage_form(
            u, voltage, v_voltage, self.experiment.I_charge, self.physical_setup)

        # DOLFINx problem and solver setup
        # ===================================
        self.solver_setup(**solver_args)

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

        # Set up a dedicated solver for the initial solve to make sure we
        # have enough iterations.
        F = [self.F, self.voltage_form]
        w = [self.u, self.voltage]

        problem = NonlinearProblemBlock(F, w)
        solver = NewtonSolver(self.mesh.comm, problem, max_iterations=50)
        self.linear_solver_setup(solver)

        its, _ = solver.solve([self.u, self.voltage])
        error = dfx.fem.petsc.assemble_vector(residual).norm()
        log(its, error)
        assert np.isclose(error, 0.)

    def solver_setup(self, **solver_args):

        F = [self.F, self.voltage_form]

        w = [self.u, self.voltage]

        problem = NonlinearProblemBlock(F, w)

        self.solver = NewtonSolver(self.mesh.comm, problem, **solver_args)

        self.linear_solver_setup(self.solver)

    def linear_solver_setup(self, solver):

        ksp = solver.ksp
        ksp.setType("preonly")
        ksp.getPC().setType("lu")

        # change below to superlu_dist for parallel computations
        ksp.getPC().setFactorSolverType("superlu")
        ksp.getPC().setFactorSetUpSolverType()

    def create_mesh(self):

        self.mesh = create_1p1_DFN_mesh(
            self.comm, self.n_radius, self.n_particles)

    def create_function_spaces(self):

        self.function_spaces = DFN_function_space(self.mesh)

    def run(self,
            t_start: float = 0.,
            t_final: float = 300.,
            n_out: int = 501,
            dt_min: float = 1e-9,
            dt_max: float = 1e-2,
            dt_increase: float = 1.1,
            tol: float = 1e-7):

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
            self.voltage,
            self.u0,
            t_final,
            self.dt,
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


def DFN_FEM_form(
    u, voltage, u0, v, dt, free_energy, Ls,
    M=lambda c: c * (1 - c), gamma=0.1, grad_c_bc=lambda c: 0.0
):

    V = u.function_space
    mesh = V.mesh
    y, mu = ufl.split(u)
    y0, mu0 = ufl.split(u0)

    v_c, v_mu = ufl.split(v)

    dA = create_particle_summation_measure(mesh)

    I_particle = - Ls * (mu + voltage)

    theta = 1.0

    c = c_of_y(y)
    c0 = c_of_y(y0)

    r, _ = ufl.SpatialCoordinate(mesh)

    s_V = 4 * np.pi * r**2
    s_A = 4 * np.pi * r**2

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


def voltage_form(u, voltage, v_voltage, I_global, physical_setup):

    y, mu = ufl.split(u)

    mesh = u.function_space.mesh

    dA = create_particle_summation_measure(mesh)

    Ls = physical_setup.reaction_affinities
    L = physical_setup.mean_affinity
    a_ratios = physical_setup.surface_weights

    num_particles = assemble_scalar(
        dfx.fem.form(dfx.fem.Constant(mesh, 1.) * ufl.dx)
    )

    F = (
        (voltage / num_particles +
         mu * Ls / L * a_ratios +
         I_global / L * a_ratios) * v_voltage * dA)

    return F
