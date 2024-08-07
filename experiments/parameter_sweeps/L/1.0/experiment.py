import os
import sys


exp_path = os.path.dirname(__file__)
parent_path = os.path.abspath(
    exp_path + "/../..")

sys.path.append(parent_path)

from default.experiment import Simulation as SimulationBase, mesh  # noqa: 402


class Simulation(SimulationBase):

    @classmethod
    def experiment(cls, *args, c_bounds=[-5., 5], **kwargs):

        super().experiment(*args, c_bounds=c_bounds, **kwargs)


if __name__ == "__main__":

    L_mean = 1.0

    simulation = Simulation(
        mesh,
        L_mean=L_mean,
        output_destination=exp_path + "/simulation_output/output")

    simulation.run(tol=1e-5)
