"""Demo file to show parallelism among functions.
"""

import dolfinx as dfx
from dolfinx.fem.petsc import LinearProblem

from mpi4py import MPI

import matplotlib.pyplot as plt

import numpy as np

import random

import ufl


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

    V = dfx.fem.FunctionSpace(mesh, ("CG", 1))

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
    c_ = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    cn = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 0.001)

    # Get an integral measure for the right boundary.
    boundaries = [(1, lambda x: np.isclose(x[0], 1.)),]

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = dfx.mesh.locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dfx.mesh.meshtags(
        mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    ds_right = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    x = ufl.SpatialCoordinate(mesh)

    # Initialize constants for particle current computation during time stepping.
    c_bc = dfx.fem.Constant(mesh, 0.)
    cell_voltage = dfx.fem.Constant(mesh, 0.)
    i_k = - L_k * (c_bc + cell_voltage)

    # An implicit Euler time step.
    residual = (c_ - cn) * v / dt * ufl.dx
    residual += ufl.dot(ufl.grad(c_), ufl.grad(v)) * ufl.dx
    residual -= i_k * v * x[0] * ds_right

    problem = LinearProblem(ufl.lhs(residual),
                            ufl.rhs(residual))

    c = dfx.fem.Function(V)
    random.seed(comm_world.rank)
    c.x.array[:] = random.random()  # <- initial data

    fig, ax = plt.subplots()

    line, = ax.plot(c.x.array[:], color=(0, 0, 0))

    it = 0
    while t < T_final:

        if comm_world.rank == 0:
            print(f"t = {t:2.4}", flush=True)

        cn.interpolate(c)

        c_bc.value = dfx.fem.assemble_scalar(dfx.fem.form(c * x[0] * ds_right))

        term = L_k * a_k * c_bc.value
        term_sum = comm_world.allreduce(term, op=MPI.SUM)

        cell_voltage.value = - (I_total.value + term_sum) / L

        c = problem.solve()

        if it % 100 == 0:

            color = (t / T_final, 0, 0)

            ax.plot(c.x.array[:], color=color)

        # line.set_ydata(c.x.array[:])
        # fig.canvas.draw()

        t += dt.value
        it += 1

    plt.show()

    if comm_world.rank == 0:
        input("Press any key to exit ...")
