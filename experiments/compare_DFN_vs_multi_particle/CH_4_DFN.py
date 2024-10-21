import dolfinx as dfx
import numpy as np
import os
from pathlib import Path

from pyMoBiMP.dfn_battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase,
    DefaultPhysicalSetup)

from pyMoBiMP.fenicsx_utils import FileOutput


class PhysicalSetup(DefaultPhysicalSetup):
    def _setup_reaction_affinity(self):

        Ls = dfx.fem.Function(self._V0)
        Ls.interpolate(
            lambda x: self.L_mean
            * (1.0 + self.L_var_rel * (2 * np.random.random(x[1].shape) - 1.0))
        )

        self._Ls = Ls


# The simulation class needs explicit input that
# specifies output, runtime_analysis, and an experiment
class Simulation(DFNSimulationBase):
    PhysicalSetup = PhysicalSetup
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment


if __name__ == "__main__":

    # Specify the simulation output directory below the
    # scipt's directory. Here, we use Path for ease of use
    # (mkdir method).
    output_dir = Path(os.path.dirname(__file__))
    output_dir /= "simulation_output/CH_4_DFN/"
    output_file = str(output_dir) + "/ch_4_dfn.xdmf"

    # Make sure the directory exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up the simulation.
    simulation = Simulation(
        n_particles=12,
        n_radius=16,
        output_destination=output_file
    )

    simulation.run(dt_max=1e-3, n_out=501, tol=1e-5, t_final=300.)
