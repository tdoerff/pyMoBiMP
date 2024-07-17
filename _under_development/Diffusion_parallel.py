"""Demo file to show parallelism among functions.
"""

import dolfinx as dfx

from mpi4py import MPI

import matplotlib.pyplot as plt

import numpy as np

import random

import ufl

from pyMoBiMP.cahn_hilliard_utils import c_of_y, _free_energy
from pyMoBiMP.fenicsx_utils import RuntimeAnalysisBase

from pyMoBiMP.fenicsx_utils import NewtonSolver, NonlinearProblem

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        comm,
        u,
        L,
        A,
        I_charge,
        c_of_y,
        free_energy,
        *args,
        filename=None,
        **kwargs
    ):
        self.comm = comm

        # TODO: Make sure this is true for more complex processor geometry.
        self.num_particles = comm.size

        self.free_energy = free_energy
        self.c_of_y = c_of_y

        self.filename = filename

        self.u_k = u

        self.L_k = L
        self.A_k = A

        self.I_charge = I_charge

        self.A = comm.allreduce(self.A_k, op=MPI.SUM)

        self.a_k = self.A_k / A

        self.L = comm.allreduce(self.L_k * self.a_k, op=MPI.SUM)

        V = self.u_k.function_space
        mesh = V.mesh

        # Pre-assemble compiled forms.
        coords = ufl.SpatialCoordinate(mesh)
        r2 = ufl.inner(coords, coords)

        y, mu = self.u_k.split()

        c = self.c_of_y(y)

        self.state_form = dfx.fem.form(3 * c * r2 * ufl.dx)
        self.mu_bc_form = dfx.fem.form(mu * r2 * ufl.ds)

        return super().setup(u, *args, **kwargs)

    def analyze(self, t):

        state_k = dfx.fem.assemble_scalar(self.state_form)

        # Altough reduce might be the right choice, there seems to
        # be a bug in the mpi4py implementation, hence we choose a
        # workaround via gather.
        states = self.comm.gather(state_k, root=0)  # all particle states

        if self.comm.rank == 0:

            # The total state of charge.
            total_state = sum(states) / self.num_particles

        # Boundary value of chemical potential
        mu_bc = dfx.fem.assemble_scalar(self.mu_bc_form)

        particle_voltage = self.L_k / self.L * self.a_k * mu_bc

        particle_voltages = self.comm.gather(particle_voltage, root=0)

        if self.comm.rank == 0:
            cell_voltage = sum(particle_voltages)
            cell_voltage += self.I_charge.value / self.L

        # if self.comm.rank == 0:
            self.data.append([total_state, cell_voltage])

            return super().analyze(t)


def log(*msg):

    print(f"[{comm_world.rank:>3}]: ", *msg, flush=True)


if __name__ == "__main__":

    # MPI communicators:
    # comm_world is the pool of all processes
    # comm_self is just the current processor.
    comm_world = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF

    # Initialize random number generator
    # ----------------------------------
    random.seed(comm_world.rank)

    # Diagnostic output
    # -----------------

    # dfx.log.set_log_level(dfx.log.LogLevel.INFO)

    # With flush=True, we force print to flush to cmd, and with barrier, we
    # make sure the first statement come first.
    if comm_world.rank == 0:
        print(f"Initialize with {comm_world.size} processes", flush=True)
    comm_world.barrier()

    print(f"Initialize on process {comm_world.rank}.", flush=True)
    comm_world.barrier()

    # Grid setup
    # ----------
    mesh = dfx.mesh.create_unit_interval(comm_self, 128)

    element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    V = dfx.fem.FunctionSpace(mesh, element * element)

    # Simulation parameters
    # ---------------------
    def M(c): return c * (1 + c)

    def free_energy(c):
        return _free_energy(c, a=0., b=0., c=0.)

    I_total = dfx.fem.Constant(mesh, 0.0)

    T_final = 1.0

    t = T_start = 0.

    R_k = 1.
    A_k = 4 * np.pi * R_k**2
    L_k = 1.e1 * (1. + 0.1 * (2 * random.random() - 1.))

    # Get global information on cell
    A = comm_world.allreduce(A_k, op=MPI.SUM)
    a_k = A_k / A

    L = comm_world.allreduce(a_k * L_k, op=MPI.SUM)

    # The FEM form
    # ------------
    u_ = dfx.fem.Function(V)
    v = ufl.TestFunction(V)

    un = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 1e-8)

    x = ufl.SpatialCoordinate(mesh)

    # Initialize constants for particle current computation during time stepping.
    i_k = dfx.fem.Constant(mesh, 0.0)

    y_, mu_ = ufl.split(u_)
    yn, mun = ufl.split(un)
    v_c, v_mu = ufl.split(v)

    y_ = ufl.variable(y_)
    c_ = c_of_y(y_)

    dcdy = ufl.diff(c_, y_)

    # Differentiate the free energy function to
    # obtain the chemical potential
    c_ = ufl.variable(c_)
    dfdc = ufl.diff(free_energy(c_), c_)
    mu_chem = dfdc

    # TODO: add geometric weights to form
    coords = ufl.SpatialCoordinate(mesh)
    r2 = ufl.dot(coords, coords)

    s_V = 4 * np.pi * r2
    s_A = 2 * np.pi * r2

    # An implicit Euler time step.
    residual = s_V * dcdy * (y_ - yn) * v_c * ufl.dx
    residual += s_V * ufl.dot(M(c_) * ufl.grad(mu_), ufl.grad(v_c)) * dt * ufl.dx
    residual -= s_A * i_k * v_c * dt * ufl.ds

    residual += (mu_ - mu_chem) * v_mu * ufl.dx

    mu_bc_form = dfx.fem.form(mu_ * r2 * ufl.ds)

    def callback(solver, uh):

        mu_bc = dfx.fem.assemble_scalar(mu_bc_form)

        weighted_particle_potential = L_k / L * a_k * mu_bc

        active_phase_potential = comm_world.allreduce(
            weighted_particle_potential, op=MPI.SUM)

        cell_voltage = -(I_total.value / L + active_phase_potential)

        i_k.value = - L_k * (mu_bc + cell_voltage)

        i_ks = comm_world.allgather(i_k.value * a_k)

        total_current = sum(i_ks)

        if not np.isclose(total_current, I_total.value):

            msg = "Partial currents do not add up to total current!\n"
            msg += f"sum(i_k a_k) = {total_current} != "
            msg += f"I_total = {I_total.value}"

            print(msg)

        return cell_voltage

    problem = NonlinearProblem(residual, u_)

    # problem.form = lambda x: callback(_, _)

    solver = NewtonSolver(comm_world, problem)
    from petsc4py import PETSc
    # solver.krylov_solver.setType(PETSc.KSP.Type.PREONLY)
    # solver.krylov_solver.getPC().setType(PETSc.PC.Type.LU)
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "gmres"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    solver.max_it = 10
    solver.rtol = 1e-3
    solver.convergence_criterion = "incremental"

    u_.sub(0).x.array[:] = -6.  # <- initial data

    if comm_world.rank == 0:
        u_.sub(0).x.array[:] = 0.33

    # Output
    # ------
    rt_analysis = AnalyzeCellPotential(
        comm_world, u_, L_k, A_k, I_total, c_of_y, free_energy,
        filename="simulation_output/Diffusion_parallel_rt.txt")

    fig, ax = plt.subplots()

    V0, _ = u_.function_space.sub(0).collapse()
    c = dfx.fem.Function(V0)

    c_expr = dfx.fem.Expression(
        c_of_y(u_.sub(0).collapse()),
        V0.element.interpolation_points())

    c.interpolate(c_expr)

    line, = ax.plot(c.x.array[:], color=(0, 0, 0))

    it = 0

    dt_min = 1e-9
    dt_max = 1e-3
    tol = 1e-7

    while t < T_final:

        # The timestep
        # ------------
        un.interpolate(u_)

        # Explicit implementation of the boundary condition
        cell_voltage = callback(solver, u_)
        # log("cell_voltage", cell_voltage, i_k.value)

        try:
            iterations, success = solver.solve(u_)
        except BaseException as e:
            log(e)

            iterations = 9e99
            success = False

        # log("before successes " + f"{success}")
        successes = comm_world.allgather(success)
        success = np.all(successes)

        if not success:
            u_.x.array[:] = un.x.array

            dt.value *= 0.5

            log(f"reduce stepsize to {dt.value:1.3e}")

            assert dt.value >= dt_min

            continue

        # Diagnostic output
        # -----------------
        iterations_global = iterations

        if comm_world.rank == 0:

            print(
                f"[{t/T_final * 100:>3.0f}%] " +
                f"t[{it:06}] = {t:2.4e} : " +
                f"dt = {dt.value:1.3e} ; " +
                f"iterations: {iterations_global} ; ",
                f"cell_voltage: {cell_voltage}", flush=True)

        # Output
        # ------
        if it % 100 == 0:

            color = (t / T_final, 0, 0)

            c_expr = dfx.fem.Expression(
                c_of_y(u_.sub(0).collapse()),
                V0.element.interpolation_points())
            c.interpolate(c_expr)

            ax.plot(c.x.array[:], color=color)

        rt_analysis.analyze(t)

        # Adaptive timestepping a la Yibao Li et al. (2017)
        u_max_loc = np.abs(u_.x.array - un.x.array).max()

        u_err_max = comm_world.allreduce(
            u_max_loc, op=MPI.MAX)

        dt.value = min(max(tol / u_err_max, dt_min),
                       dt_max,
                       1.001 * dt.value)

        t += dt.value
        it += 1

    plt.show()
