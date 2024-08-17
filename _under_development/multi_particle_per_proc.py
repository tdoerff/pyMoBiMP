# %%
import dolfinx as dfx

from mpi4py import MPI

from math import log10, ceil

import os

from pyMoBiMP.cahn_hilliard_utils import (
    MultiParticleSimulation as Simulation,
)


if __name__ == "__main__":

    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    num_of_procs = comm_world.size

    # For the output file string
    digits = ceil(log10(num_of_procs))

    def info(*msg):
        return print(f"[{rank:>{digits}}]", *msg, flush=True)

    # %%
    # Discretization
    # --------------
    mesh_comm = MPI.COMM_SELF

    n_elem = 16
    mesh = dfx.mesh.create_unit_interval(mesh_comm, n_elem)
    exp_path = os.path.dirname(os.path.abspath(__file__))

    info(f"Initialize on {num_of_procs} processes.")

    Simulation.T_final = 0.1

    simulation = Simulation(
        mesh,
        output_destination=exp_path +
        f"/simulation_output/output_{comm_world.rank:0{digits}}")

    # simulation.run(tol=1e-5)
