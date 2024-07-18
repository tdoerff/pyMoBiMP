# %%
import dolfinx as dfx

from mpi4py import MPI

import os

from pyMoBiMP.cahn_hilliard_utils import (
    MultiParticleSimulation as Simulation,
)


comm_world = MPI.COMM_WORLD


# %%
# Discretization
# --------------

n_elem = 16
mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)
exp_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    simulation = Simulation(
        mesh,
        M=lambda c: 0.1 * c * (1. - c),
        output_destination=exp_path + "/simulation_output/output")

    simulation.run(tol=1e-5)
