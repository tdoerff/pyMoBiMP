"""Demo file to show parallelism among functions.
"""

import dolfinx as dfx

from mpi4py import MPI

import matplotlib.pyplot as plt

import numpy as np

import random

import ufl

from pyMoBiMP.cahn_hilliard_utils import cahn_hilliard_form
from pyMoBiMP.cahn_hilliard_utils import _free_energy as free_energy_base
from pyMoBiMP.cahn_hilliard_utils import populate_initial_data
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
    def free_energy(c):
        return free_energy_base(c, a=0, b=0, c=0)

    I_total = dfx.fem.Constant(mesh, 0.01)

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

    y_, mu_ = ufl.split(u_)
    populate_initial_data(u_, lambda x: 1e-3 * np.ones_like(x[0]), free_energy)

    V0, dofmap_V0 = V.sub(0).collapse()

    v = ufl.TestFunction(V)

    un = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 0.001)

    x = ufl.SpatialCoordinate(mesh)

    # Initialize constants for particle current computation during time stepping.
    c_bc = dfx.fem.Constant(mesh, 0.)
    cell_voltage = dfx.fem.Constant(mesh, 0.)
    i_k = - L_k * (c_bc + cell_voltage)

    residual = cahn_hilliard_form(
        u_, un, dt, M=lambda c: (1 - c) * c,
        free_energy=free_energy,
        lam=0.1,
        I_charge=i_k
    )

    problem = NonlinearProblem(residual, u_)

    def callback(solver, u_):

        c_, _ = u_.split()

        c_bc.value = dfx.fem.assemble_scalar(dfx.fem.form(c_ * x[0] * ufl.ds))

        term = L_k * a_k * c_bc.value
        term_sum = comm_world.allreduce(term, op=MPI.SUM)

        cell_voltage.value = - (I_total.value + term_sum) / L

    solver = NewtonSolver(comm_self,
                          problem,
                          max_iterations=1000,
                          callback=callback)

    random.seed(comm_world.rank)
    u_.sub(0).x.array[:] = random.random()  # <- initial data

    fig, ax = plt.subplots()

    line, = ax.plot(u_.sub(0).collapse().x.array[:], color=(0, 0, 0))

    it = 0
    while t < T_final:

        un.interpolate(u_)

        # c = problem.solve()
        iterations, success = solver.solve(u_)

        assert success

        if comm_world.rank == 0:

            iterations = comm_world.allreduce(iterations, op=MPI.MAX)

            print(f"t = {t:2.4} : iterations: {iterations}", flush=True)

        if it % 100 == 0:

            color = (t / T_final, 0, 0)

            ax.plot(u_.sub(0).collapse().x.array[:], color=color)

        # line.set_ydata(c.x.array[:])
        # fig.canvas.draw()

        t += dt.value
        it += 1

    plt.show()

    if comm_world.rank == 0:
        input("Press any key to exit ...")
