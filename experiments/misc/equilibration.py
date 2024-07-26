# %%
import dolfinx as dfx

from mpi4py import MPI

import numpy as np

import os

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    MultiParticleSimulation as SimulationBase,
    y_of_c
)


class Simulation(SimulationBase):

    def initial_data(self):

        u_ini = dfx.fem.Function(self.V)

        y, _ = u_ini.split()

        num_particles = len(y)

        eps = 1e-3

        charges = np.linspace(eps, 1-eps, num_particles)

        # Store concentration-like quantity into state vector
        # ---------------------------------------------------

        V_c, _ = self.V.sub(0).collapse()

        c_ini = dfx.fem.Function(V_c)

        # extract number of particles
        y, _ = u_ini.split()

        num_particles = len(y.split())

        for i_particle in range(num_particles):

            c_ini.sub(i_particle).interpolate(lambda x: charges[i_particle] + 0 * x[0])

            W = c_ini.sub(i_particle).function_space
            x_interpolate = W.element.interpolation_points()

            y_ini = dfx.fem.Expression(
                y_of_c(c_ini.sub(i_particle)), x_interpolate)

            u_ini.sub(0).sub(i_particle).interpolate(y_ini)

        # Store chemical potential into state vector
        # ------------------------------------------

        for i_particle in range(num_particles):
            c_ini_ = ufl.variable(c_ini.sub(i_particle))
            dFdc1 = ufl.diff(self.free_energy(c_ini_, ufl.ln, ufl.sin), c_ini_)

            W = u_ini.sub(1).sub(i_particle).function_space
            u_ini.sub(1).sub(i_particle).interpolate(
                dfx.fem.Expression(dFdc1, W.element.interpolation_points())
            )

        u_ini.x.scatter_forward()

        return u_ini


comm_world = MPI.COMM_WORLD


if __name__ == "__main__":

    # %%
    # Discretization
    # --------------
    n_elem = 16
    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)
    exp_path = os.path.dirname(os.path.abspath(__file__))

    Simulation.T_final = 20.

    simulation = Simulation(
        mesh,
        output_destination=exp_path + "/simulation_output/equilibration",
        C_rate=0.,
        num_particles=48)

    simulation.run(tol=1e-5)
