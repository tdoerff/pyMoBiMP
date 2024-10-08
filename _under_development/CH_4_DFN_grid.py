import os
# Setting the number of OpenBLAS threads might be necessary if the OS complains
# about busy resources.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from pyMoBiMP.dfn_battery_model import (  # noqa: 402
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase,
    DefaultPhysicalSetup)

from pyMoBiMP.fenicsx_utils import FileOutput  # noqa: 402


class Simulation(DFNSimulationBase):
    PhysicalSetup = DefaultPhysicalSetup
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment


if __name__ == "__main__":

    simulation = Simulation()

    simulation.run()
