from pyMoBiMP.dfn_battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase)

from pyMoBiMP.fenicsx_utils import FileOutput


class Simulation(DFNSimulationBase):
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment


if __name__ == "__main__":

    simulation = Simulation()

    simulation.run()
