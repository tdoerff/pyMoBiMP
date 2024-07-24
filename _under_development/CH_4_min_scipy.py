import dolfinx as dfx

from mpi4py.MPI import COMM_SELF as comm

import numpy as np

import pyvista as pv

import scipy as sp

from scipy.integrate._ivp.base import OdeSolver  # this is the class we will monkey patch

from tqdm import tqdm

from pyMoBiMP.cahn_hilliard_utils import (
    get_mesh_spacing,
    SingleParticleODEProblem as ODEProblem)


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

        y = solution.y[:, it]
        c = c_of_y(y)

        chart.line(x[:, 0], c, color=(t / solution.t[-1], 0, 0))

    plotter = pv.Plotter()
    plotter.add_chart(chart)

    plotter.show()


if __name__ == "__main__":

    # Discretization
    # --------------

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm, n_elem)

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    particle_ode = ODEProblem(mesh)

    T_final = 1.

    solution = sp.integrate.solve_ivp(
        particle_ode, [0, T_final], particle_ode.y.x.array[:],
        first_step=1e-9,
        max_step=1e-3,
        min_step=1e-12,
        t_eval=np.linspace(0, T_final, 20),
        method='LSODA')

    print(solution.message)

    plot_solution(solution)
