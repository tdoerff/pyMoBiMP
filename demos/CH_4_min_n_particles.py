# %%
import dolfinx as dfx

from mpi4py import MPI

import os

from pyMoBiMP.cahn_hilliard_utils import MultiParticleSimulation


Simulation = MultiParticleSimulation


if __name__ == "__main__":

    # Execution setup
    # ---------------
    comm_world = MPI.COMM_WORLD

    # %%
    # Discretization
    # --------------

    # Set up the mesh
    mesh_filename = "Meshes/line_mesh.xdmf"

    if os.path.isfile(mesh_filename):
        # Load mesh from file
        with dfx.io.XDMFFile(comm_world, mesh_filename, 'r') as file:
            mesh = file.read_mesh(name="Grid")
    else:
        n_elem = 16
        mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    num_particles = 12

    simulation = Simulation(
        mesh,
        num_particles=num_particles,
        output_destination=f"simulation_output/CH_4_min_{num_particles}_particles")

    simulation.run(tol=1e-5)
