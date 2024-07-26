import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm_world
import os
import sys


exp_path = os.path.dirname(__file__)
parent_path = os.path.abspath(
    exp_path + "/../..")

sys.path.append(parent_path)

from default.experiment import Simulation  # noqa: 402


if __name__ == "__main__":

    # %%
    # Discretization
    # --------------

    n_elem = 128
    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    simulation = Simulation(
        mesh,
        output_destination=exp_path + "/simulation_output/output",
        gamma=1e-3)

    simulation.run(tol=1e-5)
