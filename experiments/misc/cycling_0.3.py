import os

from pyMoBiMP.battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase,
    DefaultPhysicalSetup
)

from pyMoBiMP.fenicsx_utils import (
    FileOutput
)


class Simulation(DFNSimulationBase):
    Experiment = ChargeDischargeExperiment
    PhysicalSetup = DefaultPhysicalSetup
    RuntimeAnalysis = AnalyzeOCP
    Output = FileOutput


if __name__ == "__main__":

    Simulation.Experiment.cycling = True

    c_rate = Simulation.Experiment.c_rate
    T_final = 6 / c_rate

    exp_path = os.path.dirname(__file__)

    simulation = Simulation(
        n_particles=24,
        output_destination=exp_path + "/simulation_output/cycling_0.3.xdmf")

    simulation.run(t_final=T_final)
