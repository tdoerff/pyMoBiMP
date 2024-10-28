# %%
import os

from pyMoBiMP.battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DefaultPhysicalSetup,
    DFNSimulationBase
)

from pyMoBiMP.fenicsx_utils import FileOutput


class Experiment(ChargeDischargeExperiment):

    pause = 150.0
    c_rate = 0.1

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # initialize pause timers
        self.t_pause_start = -1e99
        self.t_pause_stop = 1e99

        # Let the experiment completely control the current.
        self.status = "charging"

    def experiment(self, t, cell_voltage):

        if -cell_voltage > self.v_cell_bounds[1] and self.I_charge.value > 0.0:
            print(
                ">>> Cell voltage exceeds maximum " +
                f"(V_cell = {cell_voltage:1.3f} > {self.v_cell_bounds[1]:1.3f})."
            )

            self.t_pause_start = t
            self.t_pause_stop = t + self.pause

            self.I_charge_next_value = -1. * self.I_charge.value
            self.I_charge.value = 0.
            self.status = "paused"

        if self.status == "paused" and t >= self.t_pause_stop:

            self.I_charge.value = self.I_charge_next_value

            self.status = "discharging"

        if self.status == "discharging" and \
                cell_voltage < self.v_cell_bounds[0]:

            self.status = "stopped"
            print(">>> Stop charging.")
            self.I_charge.value = 0.0

        return False  # Always continue the simulation


class Simulation(DFNSimulationBase):
    Experiment = Experiment
    Output = FileOutput
    PhysicalSetup = DefaultPhysicalSetup
    RuntimeAnalysis = AnalyzeOCP


if __name__ == "__main__":

    c_rate = Experiment.c_rate
    T_final = 6 / c_rate + 2 * Experiment.pause

    exp_path = os.path.dirname(__file__)

    simulation = Simulation(
        n_particles=128,
        output_destination=exp_path + "/simulation_output/pause_after_charging")

    simulation.run(tol=1e-7)
