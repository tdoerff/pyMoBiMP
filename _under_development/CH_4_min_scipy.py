import basix

import dolfinx as dfx

from mpi4py.MPI import COMM_SELF as comm

import numpy as np

import pyvista as pv

import scipy as sp

from scipy.integrate._ivp.base import OdeSolver  # this is the class we will monkey patch

from tqdm import tqdm

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    _free_energy as free_energy,
    get_mesh_spacing,
    SingleParticleODEProblem,
    SingleParticleODEProblem as ODEProblem)


class MultiParticleODE(SingleParticleODEProblem):
    def __init__(self, mesh, num_particles, c_of_y, free_energy, gamma=0.1):

        # Simulation setup
        # ----------------
        self.num_particles = num_particles

        self.Ls = 1.e1 * (1. + 0.1 * (2. * np.random.random(num_particles) - 1.))
        self.a_ratios = np.ones(num_particles) / num_particles

        self.L = sum(_a * _L for _a, _L in zip(self.a_ratios, self.Ls))

        self.mesh = mesh

        # Set up function space
        # ---------------------
        element = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)

        # Set up individual particle forms
        # --------------------------------
        self.I_total = 0.1
        self.i_ks = [dfx.fem.Constant(mesh, 0.0) for _ in range(num_particles)]

        particle_odes = []

        for i_particle in range(num_particles):

            particle_odes.append(
                SingleParticleODEProblem(
                    mesh,
                    element=element,
                    c_of_y=c_of_y,
                    free_energy=free_energy,
                    gamma=gamma,
                    I_charge=self.i_ks[i_particle]
                )
            )

        self.particle_odes = particle_odes

        coords = ufl.SpatialCoordinate(mesh)
        r2 = ufl.dot(coords, coords)

        self.mu_bc_forms = [dfx.fem.form(particle_ode.y * r2 * ufl.ds)
                            for particle_ode in self.particle_odes]

    @property
    def initial_data(self):
        y_0 = np.array(
            [ode.y.x.array[:] for ode in self.particle_odes]
        ).flatten()

        return y_0

    def compute_cell_voltage(self, mu_bcs):

        weighted_particle_potentials = [
            L_k / self.L * a_k * mu_bc
            for L_k, a_k, mu_bc in zip(self.Ls, self.a_ratios, mu_bcs)
        ]

        active_phase_potential = sum(weighted_particle_potentials)

        cell_voltage = -(self.I_total / self.L + active_phase_potential)

        return cell_voltage

    def experiment(self, t):

        mu_bcs = [dfx.fem.assemble_scalar(mu_bc_form)
                  for mu_bc_form in self.mu_bc_forms]
        cell_voltage = self.compute_cell_voltage(mu_bcs)

        for i_particle, i_k in enumerate(self.i_ks):

            i_k.value = - self.Ls[i_particle] * (mu_bcs[i_particle] + cell_voltage)

    def rhs(self, t, y_vec):

        dydt = np.zeros_like(y_vec)

        stop = 0
        for i_particle in range(self.num_particles):

            # distribute the input data across the particle FEM functions
            # -----------------------------------------------------------

            # length of the current particle' function storage
            length = len(self.particle_odes[i_particle].y.x.array)

            # Start at the last stop index.
            start = stop
            # Increase the stop counter
            stop += length

            # Compute the rhs of the current particle and store it into the
            # return vector.
            particle_ode = self.particle_odes[i_particle]
            dydt[start:stop] = particle_ode(t, y_vec[start:stop])

            # TODO: make sure the boudary term i_k is computed constently at
            # t \in [t^n, t^n+1].
            self.experiment(t)

        return dydt


# From: https://towardsdatascience.com/do-stuff-at-each-ode-integration-step-monkey-patching-solve-ivp-359b39d5f2  # noqa: 501
# monkey patching the ode solvers with a progress bar

# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step


# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):

    # define the progress bar
    self.pbar = tqdm(total=t_bound - t0, unit='ut', initial=t0, ascii=True, desc='IVP')
    self.last_t = t0

    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)

    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step


def plot_solution(solution):

    cell, types, x = dfx.plot.vtk_mesh(mesh)

    chart = pv.Chart2D()

    def c_of_y(y):
        return np.exp(y) / (1. + np.exp(y))

    for it, t in enumerate(solution.t):

        y = solution.y[:len(x), it]
        c = c_of_y(y)

        chart.line(x[:, 0], c, color=(t / solution.t[-1], 0, 0))

    plotter = pv.Plotter()
    plotter.add_chart(chart)

    plotter.show()


if __name__ == "__main__":

    # General simulation setup
    # ------------------------
    num_of_particles = 2

    # Discretization
    # --------------

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm, n_elem)

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    I_charge = dfx.fem.Constant(mesh, 1.0)
    particle_ode = ODEProblem(mesh, I_charge=I_charge)

    multi_particle_ode = MultiParticleODE(
        mesh, num_of_particles, c_of_y, free_energy
    )

    T_final = 1.

    solution = sp.integrate.solve_ivp(
        multi_particle_ode, [0, T_final], multi_particle_ode.initial_data[:],
        first_step=1e-9,
        max_step=1e-3,
        min_step=1e-12,
        t_eval=np.linspace(0, T_final, 20),
        method='LSODA')

    print(solution.message)

    plot_solution(solution)
