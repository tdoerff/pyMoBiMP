import os

from pyMoBiMP.battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase,
    DefaultPhysicalSetup)

from pyMoBiMP.fenicsx_utils import FileOutput


class Simulation(DFNSimulationBase):
    PhysicalSetup = DefaultPhysicalSetup
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment


Simulation.Experiment.c_rate = 1e-2


if __name__ == "__main__":

    dir = os.path.dirname(__file__)

    output_destination = dir + "/output"

    simulation = Simulation(
        n_particles=1024,
        output_destination=output_destination)

    simulation.run(t_final=300.)
