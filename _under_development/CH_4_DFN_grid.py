import basix
import dolfinx as dfx
from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase
from dolfinx.nls.petsc import NewtonSolver

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm, SUM

import numpy as np

from petsc4py import PETSc

import pyvista as pv

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    compute_chemical_potential,
    _free_energy as free_energy)

from pyMoBiMP.fenicsx_utils import (
    FileOutput,
    RuntimeAnalysisBase,
    StopEvent)

# %% Helper functions
# ===================


def time_stepping_iterate_over_voltage(
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

    # Prepare outout
    if output is not None:
        output = np.atleast_1d(output)

    it = 0

    while t < T:

        it += 1

        if runtime_analysis is not None:
            runtime_analysis.analyze(t)

        try:
            u.x.scatter_forward()
            u0.x.array[:] = u.x.array[:]
            u0.x.scatter_forward()

            V_cell.update()
            callback()  # check current
            stop = event_handler(t, cell_voltage=V_cell.value, **event_pars)

            if stop:
                break

            if float(dt) < dt_min:

                raise ValueError(f"Timestep too small (dt={dt.value})!")

            tol_voltage = 1e-6
            voltage_it_max = 20
            it_voltage = 0

            inc_voltage = 1e99  # To enter the loop at least once.

            V_cell_value_old = V_cell.value

            pass

            while inc_voltage > tol_voltage and \
                    it_voltage <= voltage_it_max:

                iterations, success = solver.solve(u)

                V_cell.update()  # recompute voltage.
                callback()

                inc_voltage = np.abs(V_cell.value - V_cell_value_old)
                V_cell_value_old = V_cell.value

                it_voltage += 1
            else:
                success = inc_voltage < tol_voltage

            if not success:
                raise RuntimeError("Newton solver did not converge.")

            # Adaptive timestepping a la Yibao Li et al. (2017)
            u_max_loc = np.abs(u.sub(0).x.array - u0.sub(0).x.array).max()

            u_err_max = u.function_space.mesh.comm.allreduce(u_max_loc, op=MPI.MAX)

            if it_voltage < voltage_it_max / 5:
                # Use the given increment factor if we are in a safe region, i.e.,
                # if the Newton solver converges sufficiently fast.
                inc_factor = dt_increase
            elif it_voltage < voltage_it_max / 2:
                # Reduce the increment if we take more iterations.
                inc_factor = 1 + 0.1 * (dt_increase - 1.)
            elif it_voltage > voltage_it_max * 0.8:
                # Reduce the timestep in case we are approaching max_it
                inc_factor = 0.9
            else:
                # Do not increase timestep between [0.5*max_it, 0.8*max_it]
                inc_factor = 1.0

            dt.value = min(max(tol / u_err_max, dt_min),
                           dt_max,
                           inc_factor * dt.value)

        except StopEvent as e:

            print(e)
            print(">>> Stop integration.")

            break

        except RuntimeError as e:

            print(e)

            # reset and continue with smaller time step.
            u.x.array[:] = u0.x.array[:]

            iterations = solver.max_it

            if dt.value > dt_min:
                dt.value *= 0.5

                print(f"Decrease timestep to dt={dt.value:1.3e}")

                continue

            else:
                if output is not None:

                    [o.save_snapshot(u, t) for o in output]

        except ValueError as e:

            print(e)

            if output is not None:
                [o.save_snapshot(u, t, force=True) for o in output]

            break

        # Find the minimum timestep among all processes.
        # Note that we explicitly use COMM_WORLD since the mesh communicator
        # only groups the processes belonging to one particle.
        dt_global = MPI.COMM_WORLD.allreduce(dt.value, op=MPI.MIN)

        dt.value = dt_global

        t += float(dt)

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
                    f"its_V = {it_voltage}",
                    flush=True
                )

    else:

        if output is not None:

            [o.finalize() for o in output]

    return


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

    # Prepare outout
    if output is not None:
        output = np.atleast_1d(output)

    it = 0

    while t < T:

        it += 1

        if runtime_analysis is not None:
            runtime_analysis.analyze(t)

        try:
            u.x.scatter_forward()
            u0.x.array[:] = u.x.array[:]
            u0.x.scatter_forward()

            V_cell.update()
            callback()  # check current
            stop = event_handler(t, cell_voltage=V_cell.value, **event_pars)

            if stop:
                break

            if float(dt) < dt_min:

                raise ValueError(f"Timestep too small (dt={dt.value})!")

            iterations, success = solver.solve(u)

            if not success:
                raise RuntimeError("Newton solver did not converge.")

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

            dt.value = min(max(tol / u_err_max, dt_min),
                           dt_max,
                           inc_factor * dt.value)

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
                dt.value *= 0.5

                print(f"Decrease timestep to dt={dt.value:1.3e}")

                continue

            else:
                if output is not None:

                    [o.save_snapshot(u, t) for o in output]

        except ValueError as e:

            print(e)

            if output is not None:
                [o.save_snapshot(u, t, force=True) for o in output]

            break

        # Find the minimum timestep among all processes.
        # Note that we explicitly use COMM_WORLD since the mesh communicator
        # only groups the processes belonging to one particle.
        dt_global = MPI.COMM_WORLD.allreduce(dt.value, op=MPI.MIN)

        dt.value = dt_global

        t += float(dt)

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

    plotter.add_mesh(warped, show_edges=True, show_vertices=False, )
    plotter.add_axes()

    plotter.show()


class Voltage(dfx.fem.Constant):

    def __init__(self,
                 u: dfx.fem.Function,
                 I_global: float | dfx.fem.Constant):

        self.u = u
        self.function_space = u.function_space
        self.I_global = I_global

        dA_R = create_particle_summation_measure(self.function_space.mesh)
        A, a_ratios, L, Ls = physical_setup(self.function_space)

        self.L = L
        self.Ls = Ls
        self.a_ratios = a_ratios
        self.A = A

        _, mu = ufl.split(u)

        V_cell_ufl = - mu * Ls / L * a_ratios * dA_R
        V_cell_ufl -= I_global / L * a_ratios * dA_R

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
    def __init__(self, u, V_cell, I_global):

        _, mu = ufl.split(u)

        a_ratios = V_cell.a_ratios
        Ls = V_cell.Ls

        dA = create_particle_summation_measure(u.function_space.mesh)

        I_particle = - Ls * (mu + V_cell)

        I_global_ref_ufl = a_ratios * I_particle * dA

        self.I_global_ref_form = dfx.fem.form(I_global_ref_ufl)
        self.I_global = I_global
        self.V_cell = V_cell

    def compute_current(self):
        self.V_cell.update()
        I_global_ref = dfx.fem.assemble_scalar(self.I_global_ref_form)
        I_global_ref = comm.allreduce(I_global_ref, op=SUM)

        return I_global_ref

    def __call__(self):

        I_global_ref = self.compute_current()

        if not np.isclose(I_global_ref, self.I_global.value):
            raise AssertionError(
                "Error in global current computation" +
                f"I_global_ref = {I_global_ref} != {self.I_global.value} = I_global.value")

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

        # Create integral measure on the particle surface
        fdim = mesh.topology.dim - 1

        facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

        facet_markers = np.full_like(facets, 1)

        facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

        dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
        dA_R = dA(1)

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


def create_1p1_DFN_mesh(comm, n_rad=16, n_part=192):

    # Nodes
    # -----
    radial_grid = np.linspace(0, 1, n_rad)
    particle_grid = np.linspace(0, 1, n_part)

    rr, pp = np.meshgrid(radial_grid, particle_grid)

    coords_grid = np.stack((rr, pp)).transpose((-1, 1, 0)).copy()

    if comm.rank == 0:
        coords_grid_flat = coords_grid.reshape(-1, 2).copy()
    else:
        coords_grid_flat = np.empty((0, 2), dtype=np.float64)

    # Elements
    # --------
    # All the radial connections
    elements_radial = [
        [[n_part * i + k, n_part * (i + 1) + k] for i in range(n_rad - 1)]
        for k in range(n_part)
    ]

    elements_radial = np.array(elements_radial).reshape(-1, 2)

    # Connections between particles
    elements_bc = (n_rad - 1) * n_part + np.array([[k, k + 1] for k in range(n_part - 1)])
    elements_bc = []  # With elements at the outer edge the integration fails.

    if comm.rank == 0:
        elements = np.array(list(elements_bc) + list(elements_radial))
    else:
        elements = np.empty((0, 2), dtype=np.int64)

    # %% The DOLFINx grid
    # -------------------

    gdim = coords_grid_flat.shape[1]
    shape = "interval"
    degree = 1

    domain = ufl.Mesh(
        basix.ufl.element("Lagrange",
                          shape,
                          degree,
                          shape=(gdim,)))

    mesh = dfx.mesh.create_mesh(comm,
                                elements[:, :gdim],
                                coords_grid_flat,
                                domain)

    return mesh


def DFN_function_space(mesh):
    # %% The DOLFINx function space
    # -----------------------------
    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    V = dfx.fem.functionspace(mesh, basix.ufl.mixed_element([elem1, elem1]))

    return V


def create_particle_summation_measure(mesh):
    # %% Create integral measure on the particle surface
    # --------------------------------------------------
    fdim = mesh.topology.dim - 1

    facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

    facet_markers = np.full_like(facets, 1)

    facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
    dA_R = dA(1)

    return dA_R


def physical_setup(V):

    mesh = V.mesh

    dA_R = create_particle_summation_measure(mesh)

    V0, _ = V.sub(0).collapse()  # <- auxiliary space for coefficient functions

    # %% Physical setup of the problem
    # ================================

    # particle parameters
    Rs = dfx.fem.Function(V0)
    Rs.x.array[:] = 1.
    Rs.x.scatter_forward()

    As = 4 * np.pi * Rs**2

    # Reaction affinity of the particles.
    L_mean = 10.
    L_var_rel = 0.1

    Ls = dfx.fem.Function(V0)
    Ls.x.array[:] = L_mean + \
        L_var_rel * (2 * np.random.random(Ls.x.array.shape) - 1)

    A_ufl = As * dA_R
    A = dfx.fem.assemble_scalar(dfx.fem.form(A_ufl))
    A = mesh.comm.allreduce(A, op=SUM)

    a_ratios = As / A

    # Weighted mean reaction affinity parameter taken from particle surfaces.
    L_ufl = a_ratios * Ls * dA_R
    L = dfx.fem.assemble_scalar(dfx.fem.form(L_ufl))
    L = mesh.comm.allreduce(L, op=SUM)

    return A, a_ratios, L, Ls


def DFN_FEM_form(
    u, u0, v, dt, I_global, V_cell,
    M=lambda c: c * (1 - c), lam=0.1, grad_c_bc=lambda c: 0.0
):

    V = u.function_space
    mesh = V.mesh
    y, mu = ufl.split(u)
    y0, mu0 = ufl.split(u0)

    v_c, v_mu = ufl.split(v)

    dA_R = create_particle_summation_measure(mesh)
    Ls = V_cell.Ls

    I_particle = - Ls * (mu + V_cell)

    theta = 1.0

    c = c_of_y(y)

    r, _ = ufl.SpatialCoordinate(mesh)

    s_V = 4 * np.pi * r**2
    s_A = 2 * np.pi * r**2

    dx = ufl.dx  # The volume element

    mu_chem = compute_chemical_potential(free_energy, c)
    mu_theta = theta * mu + (theta - 1.0) * mu0

    flux = M(c) * mu_theta.dx(0)

    # %% The FEM form
    # ===============
    F1 = s_V * (c_of_y(y) - c_of_y(y0)) * v_mu * dx
    F1 += s_V * flux * v_mu.dx(0) * dt * dx
    F1 -= I_particle * s_A * v_mu * dt * dA_R

    F2 = s_V * mu * v_c * dx
    F2 -= s_V * mu_chem * v_c * dx
    F2 -= lam * (s_V * c.dx(0) * v_c.dx(0) * dx)
    F2 += grad_c_bc(c) * (s_A * v_c * dA_R)

    F = F1 + F2

    return F


if __name__ == "__main__":

    mesh = create_1p1_DFN_mesh(comm, n_rad=16, n_part=256)

    V = DFN_function_space(mesh)

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    v = ufl.TestFunction(V)

    # Initial data
    u.sub(0).x.array[:] = -6.90675478  # This corresponds to the leftmost minimum of F

    dt_min = 1e-9
    dt_max = 1e1

    dt = dfx.fem.Constant(mesh, 1e-8)

    T_final = 650.0

    I_global = dfx.fem.Constant(mesh, 0.01)
    V_cell = Voltage(u, I_global)

    # FEM Form
    # ========
    F = DFN_FEM_form(u, u0, v, dt, I_global, V_cell)

    # %% Runtime analysis and output
    # ==============================
    rt_analysis = AnalyzeOCP(u,
                             c_of_y,
                             V_cell,
                             filename="CH_4_DFN_rt.txt")

    output = FileOutput(u,
                        np.linspace(0, T_final, 101),
                        filename="CH_4_DFN.xdmf",
                        variable_transform=c_of_y)

    callback = TestCurrent(u, V_cell, I_global)

    # %% DOLFINx problem and solver setup
    # ===================================

    problem = NonlinearProblem(F, u, callback=callback)
    solver = NewtonSolver(comm, problem)
    solver.rtol = 1e-9
    solver.max_it = 50
    solver.convergence_criterion = "incremental"
    solver.relaxation_parameter = 1.0

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    # %% Run the simulation
    # =====================
    time_stepping(
        solver,
        u,
        u0,
        T_final,
        dt,
        V_cell,
        dt_max=dt_max,
        dt_min=dt_min,
        dt_increase=1.1,
        tol=1e-5,
        runtime_analysis=rt_analysis,
        output=output,
        callback=callback
    )
