import os
import sys


exp_path = os.path.dirname(__file__)
parent_path = os.path.abspath(
    exp_path + "/../..")

sys.path.append(parent_path)

from default.experiment import Simulation, mesh  # noqa: 402


if __name__ == "__main__":

    simulation = Simulation(
        mesh,
        output_destination=exp_path + "/simulation_output/output",
        num_particles=1)

    simulation.run(tol=1e-5)
