# %%
import dolfinx as dfx

from mpi4py import MPI

import os

from pyMoBiMP.cahn_hilliard_utils import (
    ChargeDischargeExperiment as ExperimentBase,
    MultiParticleSimulation as Simulation,
)


comm_world = MPI.COMM_WORLD


class Experiment(ExperimentBase):

    pause = 150.0
    charge_amplitude = 0.1

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # initialize pause timers
        self.t_pause_start = -1e99
        self.t_pause_stop = 1e99

        # Let the experiment completely control the current.
        self.status = "charging"
        self.I_charge.value = self.charge_amplitude

    def experiment(self, t, cell_voltage):

        if cell_voltage > self.c_bounds[1] and self.I_charge.value > 0.0:
            print(
                ">>> Cell voltage exceeds maximum " +
                f"(V_cell = {cell_voltage:1.3f} > {self.c_bounds[1]:1.3f})."
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
                cell_voltage < self.c_bounds[0]:

            self.status = "stopped"
            print(">>> Stop charging.")
            self.I_charge.value = 0.0

        return False  # Always continue the simulation


if __name__ == "__main__":

    # %%
    # Discretization
    # --------------
    n_elem = 16
    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)
    exp_path = os.path.dirname(os.path.abspath(__file__))

    Simulation.Experiment = Experiment

    c_rate = 3 * Experiment.charge_amplitude
    T_final = 6 / c_rate + 2 * Experiment.pause

    Simulation.T_final = T_final

    simulation = Simulation(
        mesh,
        output_destination=exp_path + "/simulation_output/pause_after_charging")

    simulation.run(tol=1e-5, dt_max=1e-3)
