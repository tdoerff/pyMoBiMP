import dolfinx as dfx

from mpi4py import MPI

import numpy as np

import os

import ufl

from pyMoBiMP.cahn_hilliard_utils import y_of_c

from CH_4_min_n_particles import MultiParticleSimulation as SimulationBase


class Simulation(SimulationBase):

    @staticmethod
    def free_energy(u, log, sin):
        return (
            u * log(u)
            + (1 - u) * log(1 - u)
        )

    def initial_data(self):
        # Balanced state for initial data.
        eps = 1e-2

        c_left = 1e-3

        u_ini = dfx.fem.Function(self.V)

        # Constant
        def c_ini_fun(x):
            return eps * np.ones_like(x[0])

        # Store concentration-like quantity into state vector
        # ---------------------------------------------------

        V_c, _ = self.V.sub(0).collapse()

        c_ini = dfx.fem.Function(V_c)

        # extract number of particles
        y, _ = u_ini.split()

        num_particles = len(y.split())

        for i_particle in range(num_particles):

            c_ini.sub(i_particle).interpolate(lambda x: c_left + 0 * c_ini_fun(x))

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


if __name__ == "__main__":

    comm_world = MPI.COMM_WORLD

    # Set up the mesh
    mesh_filename = "Meshes/line_mesh.xdmf"

    if os.path.isfile(mesh_filename):
        # Load mesh from file
        with dfx.io.XDMFFile(comm_world, mesh_filename, 'r') as file:
            mesh = file.read_mesh(name="Grid")
    else:
        n_elem = 16
        mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    simulation = Simulation(
        mesh,
        output_destination="simulation_output/Diff_10_particles")

    simulation.run()
