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

    @staticmethod
    def mobility(c):
        return 0.01 * c * (1 - c)


Simulation.Experiment.c_rate = 1e-2


if __name__ == "__main__":

    dir = os.path.dirname(__file__)

    output_destination = dir + "/output"

    simulation = Simulation(
        n_particles=256,
        output_destination=output_destination)

    simulation.run(dt_max=1e-3, tol=1e-6, t_final=300.)
