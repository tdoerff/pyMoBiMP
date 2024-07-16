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


class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        comm,
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

        self.free_energy = free_energy
        self.c_of_y = c_of_y

        self.filename = filename

        self.L_k = L
        self.A_k = A

        self.I_charge = I_charge

        self.A = comm.allreduce(self.A_k, op=MPI.SUM)

        self.a_k = self.A_k / A

        self.L = comm.allreduce(self.L_k * self.a_k, op=MPI.SUM)

        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):

        V = u_state.function_space
        mesh = V.mesh

        y, mu = u_state.split()

        c = self.c_of_y(y)

        # TODO: this can be done at initialization.
        coords = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(sum([co**2 for co in coords]))

        charge_k = dfx.fem.assemble_scalar(dfx.fem.form(3 * c * r**2 * ufl.dx))

        charge = self.comm.allreduce(charge_k, op=MPI.SUM)

        mu_bc = dfx.fem.assemble_scalar(dfx.fem.form(mu * r**2 * ufl.ds))

        particle_voltage = self.L_k / self.L * self.a_k * mu_bc

        cell_voltage = self.comm.allreduce(particle_voltage, op=MPI.SUM)
        cell_voltage += self.I_charge.value / self.L

        self.data.append([charge, cell_voltage])

        return super().analyze(u_state, t)


if __name__ == "__main__":

    # MPI communicators:
    # comm_world is the pool of all processes
    # comm_self is just the current processor.
    comm_world = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF

    # Diagnostic output
    # -----------------

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

    I_total = dfx.fem.Constant(mesh, 0.1)

    T_final = 1.0

    t = T_start = 0.

    R_k = 1.
    A_k = 4 * np.pi * R_k**2
    L_k = 1.e1 * (1 + 0.1 * (2 * random.random() - 1))

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
    mu_bc = dfx.fem.Constant(mesh, 0.)
    cell_voltage = dfx.fem.Constant(mesh, 0.)
    i_k = - L_k * (mu_bc + cell_voltage)

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
    r = sum(co**2 for co in coords)

    s_V = 4 * np.pi * r**2
    s_A = 2 * np.pi * r**2

    # An implicit Euler time step.
    residual = s_V * dcdy * (y_ - yn) * v_c / dt * ufl.dx
    residual += s_V * ufl.dot(M(c_) * ufl.grad(mu_), ufl.grad(v_c)) * ufl.dx
    residual -= s_A * I_total * v_c * ufl.ds

    residual += (mu_ - mu_chem) * v_mu * ufl.dx

    problem = NonlinearProblem(residual, u_)

    def callback(solver, uh):

        _, muh = uh.split()

        mu_bc.value = dfx.fem.assemble_scalar(
            dfx.fem.form(muh * r**2 * ufl.ds))

        term = L_k * a_k * mu_bc.value
        term_sum = comm_world.allreduce(term, op=MPI.SUM)

        cell_voltage.value = - (I_total.value + term_sum) / L

    solver = NewtonSolver(comm_self,
                          problem,
                          max_iterations=1000,
                          callback=callback)

    u_.sub(0).x.array[:] = -6. * comm_world.rank  # <- initial data

    # Output
    # ------
    rt_analysis = AnalyzeCellPotential(
        comm_world, L_k, A_k, I_total, c_of_y, free_energy,
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
    tol = 1e-4

    while t < T_final:

        # The timestep
        # ------------
        un.interpolate(u_)

        callback(solver, u_)  # <- this means the boundary conditions is explicit

        iterations, success = solver.solve(u_)

        assert success

        # Diagnostic output
        # -----------------
        iterations_global = comm_world.allreduce(iterations, op=MPI.MAX)

        if comm_world.rank == 0:

            print(f"t = {t:2.4} : " +
                  f"dt = {dt.value:1.3e} ; " +
                  f"iterations: {iterations_global}", flush=True)

        # Output
        # ------
        if it % 100 == 0:

            color = (t / T_final, 0, 0)

            c_expr = dfx.fem.Expression(
                c_of_y(u_.sub(0).collapse()),
                V0.element.interpolation_points())
            c.interpolate(c_expr)

            ax.plot(c.x.array[:], color=color)

        rt_analysis.analyze(u_, t)

        # Adaptive timestepping a la Yibao Li et al. (2017)
        u_max_loc = np.abs(u_.x.array - un.x.array).max()

        u_err_max = comm_world.allreduce(
            u_max_loc, op=MPI.MAX)

        dt.value = min(max(tol / u_err_max, dt_min),
                       dt_max,
                       1.1 * dt.value)

        t += dt.value
        it += 1

    plt.show()
