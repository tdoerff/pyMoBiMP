"""Demo file to show parallelism among functions.
"""

import dolfinx as dfx

from mpi4py import MPI

import matplotlib.pyplot as plt

from mpi4py import MPI

import numpy as np

import random

import ufl

from pyMoBiMP.cahn_hilliard_utils import cahn_hilliard_form
from pyMoBiMP.cahn_hilliard_utils import c_of_y
from pyMoBiMP.cahn_hilliard_utils import _free_energy as free_energy_base
from pyMoBiMP.cahn_hilliard_utils import populate_initial_data
from pyMoBiMP.cahn_hilliard_utils import RuntimeAnalysisBase
from pyMoBiMP.fenicsx_utils import NewtonSolver, NonlinearProblem, time_stepping


class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        comm,
        L,
        A,
        I_charge,
        *args,
        c_of_y,
        free_energy=free_energy_base,
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

    coords = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(sum([co**2 for co in coords]))

    element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = dfx.fem.FunctionSpace(mesh, element * element)

    V0 = dfx.fem.FunctionSpace(mesh, element)

    # Simulation parameters
    # ---------------------
    def free_energy(c):
        return free_energy_base(c)

    I_total = dfx.fem.Constant(mesh, 0.01)

    T_final = 0.1

    t = T_start = 0.

    R_k = 1.
    A_k = 4 * np.pi * R_k**2
    L_k = 1.e1 * (1 + 0.1 * (2 * random.random() - 1))

    # Get global information on cell
    A = comm_world.allreduce(A_k, op=MPI.SUM)  # total surface
    a_k = A_k / A  # partial surface of current particle

    L = comm_world.allreduce(a_k * L_k, op=MPI.SUM)

    # The FEM form
    # ------------
    u_ = dfx.fem.Function(V)

    c_ = dfx.fem.Function(V0)  # For plotting purposes.

    y_, mu_ = ufl.split(u_)
    populate_initial_data(u_, lambda x: 1e-3 * np.ones_like(x[0]), free_energy)

    v = ufl.TestFunction(V)

    un = dfx.fem.Function(V)
    un.interpolate(u_)

    dt = dfx.fem.Constant(mesh, 1e-5)

    # Initialize constants for particle current computation during time stepping.
    i_k = dfx.fem.Constant(mesh, 0.1)

    residual = cahn_hilliard_form(
        u_, un, dt,
        M=lambda c: (1 - c) * c,
        c_of_y=c_of_y,
        free_energy=free_energy,
        lam=0.1,
        I_charge=i_k,
        theta=1.0
    )

    problem = NonlinearProblem(residual, u_)

    def callback(_, u):

        _, mu_ = u.split()

        mu_bc = dfx.fem.assemble_scalar(dfx.fem.form(mu_ * r**2 * ufl.ds))

        term = L_k * a_k * mu_bc
        term_sum = comm_world.allreduce(term, op=MPI.SUM)

        cell_voltage = (I_total.value + term_sum) / L

        i_k.value = L_k * (-mu_bc + cell_voltage)

        i_sum = comm_world.allreduce(i_k.value * a_k, op=MPI.SUM)

        if abs(i_sum - I_total.value) > 1e-6:
            raise RuntimeError(
                "partial currents do not add up to total current: " +
                f"{i_sum} != {I_total.value}")

        # print(comm_world.rank, i_k.value, i_sum, mu_bc, flush=True)

    solver = NewtonSolver(comm_self,
                          problem,
                          max_iterations=100,
                          callback=callback)

    fig, ax = plt.subplots()

    rt_analysis = AnalyzeCellPotential(
        comm_world, L_k, A_k, I_total, c_of_y=c_of_y,
        filename="simulation_output/Diffusion_parallel_rt.txt")

    def callback(it, t, u):

        if it % 100 == 0:

            y = u.sub(0).collapse()

            c_expr = dfx.fem.Expression(c_of_y(y),
                                        V0.element.interpolation_points())
            c_.interpolate(c_expr)

            ax.plot(c_.x.array)

    time_stepping(solver, u_, un, T_final, dt,
                  runtime_analysis=rt_analysis,
                  callback=callback)

    plt.show()
