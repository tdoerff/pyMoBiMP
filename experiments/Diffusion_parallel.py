"""Demo file to show parallelism among functions.
"""

import dolfinx as dfx

from mpi4py import MPI

import matplotlib.pyplot as plt

import numpy as np

import random

import ufl

from pyMoBiMP.fenicsx_utils import NewtonSolver, NonlinearProblem


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
    I_total = dfx.fem.Constant(mesh, 1.0)

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

    dt = dfx.fem.Constant(mesh, 0.001)

    x = ufl.SpatialCoordinate(mesh)

    # Initialize constants for particle current computation during time stepping.
    c_bc = dfx.fem.Constant(mesh, 0.)
    cell_voltage = dfx.fem.Constant(mesh, 0.)
    i_k = - L_k * (c_bc + cell_voltage)

    c_, mu_ = ufl.split(u_)
    cn, mun = ufl.split(un)
    v_c, v_mu = ufl.split(v)

    # An implicit Euler time step.
    residual = (c_ - cn) * v_c / dt * ufl.dx
    residual += ufl.dot(ufl.grad(c_), ufl.grad(v_c)) * ufl.dx
    residual -= i_k * v_c * x[0] * ufl.ds

    residual += (mu_ - c_) * v_mu * ufl.dx

    problem = NonlinearProblem(residual, u_)

    def callback(solver, uh):

        _, muh = uh.split()

        c_bc.value = dfx.fem.assemble_scalar(dfx.fem.form(muh * x[0] * ufl.ds))

        term = L_k * a_k * c_bc.value
        term_sum = comm_world.allreduce(term, op=MPI.SUM)

        cell_voltage.value = - (I_total.value + term_sum) / L

    solver = NewtonSolver(comm_self,
                          problem,
                          max_iterations=1000)

    random.seed(comm_world.rank)
    u_.sub(0).x.array[:] = random.random()  # <- initial data

    fig, ax = plt.subplots()

    V0, _ = u_.function_space.sub(0).collapse()
    c = dfx.fem.Function(V0)

    c.interpolate(u_.sub(0))

    line, = ax.plot(c.x.array[:], color=(0, 0, 0))

    it = 0

    dt_min = 1e-9
    dt_max = 1e-2
    tol = 1e-4

    while t < T_final:

        un.interpolate(u_)

        callback(solver, u_)  # <- this means the boundary conditions is explicit

        # c = problem.solve()
        iterations, success = solver.solve(u_)

        assert success

        iterations = comm_world.allreduce(iterations, op=MPI.MAX)

        if comm_world.rank == 0:

            print(f"t = {t:2.4} : iterations: {iterations}", flush=True)

        if it % 100 == 0:

            color = (t / T_final, 0, 0)

            c.interpolate(u_.sub(0))

            ax.plot(c.x.array[:], color=color)

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

    if comm_world.rank == 0:
        input("Press any key to exit ...")
