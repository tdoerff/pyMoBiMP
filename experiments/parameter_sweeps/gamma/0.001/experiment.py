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

    # Set up the mesh
    mesh_filename = parent_path + "/../../" + "Meshes/line_mesh.xdmf"

    if os.path.isfile(mesh_filename):
        # Load mesh from file
        with dfx.io.XDMFFile(comm_world, mesh_filename, 'r') as file:
            mesh = file.read_mesh(name="Grid")

    simulation = Simulation(
        mesh,
        output_destination=exp_path + "/simulation_output/output",
        gamma=1e-3)

    simulation.run(tol=1e-6, dt_max=1e-3)
