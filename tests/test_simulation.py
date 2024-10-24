import pytest

from pyMoBiMP.battery_model import (
    DefaultPhysicalSetup,
    DFNSimulationBase,
    AnalyzeOCP,
    ChargeDischargeExperiment)

from pyMoBiMP.fenicsx_utils import FileOutput


def test_instatiate_default():

    with pytest.raises(NotImplementedError):
        class SimulationBase(DFNSimulationBase):
            pass

        SimulationBase()

    with pytest.raises(NotImplementedError):
        class SimulationNoPhysicalSetup(DFNSimulationBase):
            # PhysicalSetup = DefaultPhysicalSetup
            RuntimeAnalysis = AnalyzeOCP
            Experiment = ChargeDischargeExperiment
            Output = FileOutput

        SimulationNoPhysicalSetup()

    with pytest.raises(NotImplementedError):
        class SimulationNoRuntimeAnalysis(DFNSimulationBase):
            PhysicalSetup = DefaultPhysicalSetup
            # RuntimeAnalysis = AnalyzeOCP,
            Experiment = ChargeDischargeExperiment
            Output = FileOutput

        SimulationNoRuntimeAnalysis()

    with pytest.raises(NotImplementedError):
        class SimulationNoExperiment(DFNSimulationBase):
            PhysicalSetup = DefaultPhysicalSetup
            RuntimeAnalysis = AnalyzeOCP
            # Experiment = ChargeDischargeExperiment
            Output = FileOutput

        SimulationNoExperiment()

    with pytest.raises(NotImplementedError):
        class SimulationNoOutput(DFNSimulationBase):
            PhysicalSetup = DefaultPhysicalSetup
            RuntimeAnalysis = AnalyzeOCP
            Experiment = ChargeDischargeExperiment
            # Output = FileOutput

        SimulationNoOutput()

    class Simulation(DFNSimulationBase):
        PhysicalSetup = DefaultPhysicalSetup
        RuntimeAnalysis = AnalyzeOCP
        Experiment = ChargeDischargeExperiment
        Output = FileOutput

    Simulation()


def test_run_simulation():

    class Simulation(DFNSimulationBase):
        PhysicalSetup = DefaultPhysicalSetup
        RuntimeAnalysis = AnalyzeOCP
        Experiment = ChargeDischargeExperiment
        Output = FileOutput

    simulation = Simulation(n_particles=2)

    simulation.run(t_final=1.)
