# %%
import dolfinx as dfx

from mpi4py import MPI

import os

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    ChargeDischargeExperiment as ExperimentBase,
    MultiParticleSimulation as Simulation,
)


comm_world = MPI.COMM_WORLD


class Experiment(ExperimentBase):

    charge_amplitude = 0.01

    c_bounds = [0.05, 0.4]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Let the experiment completely control the current.
        self.status = "charging"
        self.I_charge.value = self.charge_amplitude

        # Initialize tracker for SoC
        y, _ = self.u.split()
        self.num_particles = len(y)

        mesh = self.u.function_space.mesh
        coords = ufl.SpatialCoordinate(mesh)
        r_square = ufl.inner(coords, coords)

        cs = [self.c_of_y(y_) for y_ in y]
        self.cs_forms = [dfx.fem.form(3 * r_square * c_ * ufl.dx) for c_ in cs]

    def experiment(self, t, cell_voltage):

        soc = sum(
            [dfx.fem.assemble_scalar(c_form) for c_form in self.cs_forms]) /\
                self.num_particles

        if soc > self.c_bounds[1] and self.I_charge.value > 0.0:
            print(
                ">>> SoC voltage exceeds maximum " +
                f"(SoC = {soc:1.3f} > {self.c_bounds[1]:1.3f})."
            )
            self.I_charge.value = -self.charge_amplitude

        if soc < self.c_bounds[0] and self.I_charge.value < 0.0:
            print(
                ">>> SoC voltage exceeds minimum " +
                f"(SoC = {soc:1.3f} > {self.c_bounds[0]:1.3f})."
            )
            self.I_charge.value = self.charge_amplitude

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
    T_final = 6 / c_rate

    Simulation.T_final = T_final

    simulation = Simulation(
        mesh,
        num_particles=24,
        output_destination=exp_path + "/simulation_output/cycling_0.3.xdmf")

    simulation.run(tol=1e-5, dt_max=1e-3)
