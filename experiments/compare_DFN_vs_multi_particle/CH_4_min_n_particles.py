# %%
import dolfinx as dfx

from mpi4py import MPI

import os

from pathlib import Path

from pyMoBiMP.cahn_hilliard_utils import MultiParticleSimulation


Simulation = MultiParticleSimulation


if __name__ == "__main__":

    # Execution setup
    # ---------------
    comm_world = MPI.COMM_WORLD

    # %%
    # Discretization
    # --------------
    n_elem = 16
    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    num_particles = 12

    output_dir = Path(os.path.dirname(__file__))
    output_dir /= f"simulation_output/CH_4_{num_particles}/"
    output_file = str(output_dir) + f"/ch_4_{num_particles}.xdmf"

    output_dir.mkdir(exist_ok=True, parents=True)

    simulation = Simulation(
        mesh,
        num_particles=num_particles,
        output_destination=output_file,
        n_out=501)

    simulation.run(tol=1e-5, dt_max=1e-3)
