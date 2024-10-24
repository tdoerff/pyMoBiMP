import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm

import numpy as np

import os

import scifem

import ufl

from pyMoBiMP.fenicsx_utils import (
    read_data, NonlinearProblemBlock, NewtonSolver)


def test_read_data_DFN_shape():

    n_particles = 2
    n_radius = 16

    directory = os.path.dirname(__file__) + "/data/DFN_simulation/"
    basename = "output"

    num_particles, t, x_data, u_data, rt_data = \
        read_data(directory + basename)

    assert num_particles == n_particles

    # In case of a DFN mesh we get two arrays of dimension [particles, radius]
    assert x_data.shape == (n_radius, )
    assert u_data.shape == (len(t), n_particles, 2, n_radius)


def test_read_data_multi_particle_shape():

    n_particles = 12
    n_radius = 16

    directory = os.path.dirname(__file__) + "/data/multi_particle_simulation/"
    basename = "output"

    num_particles, t, x_data, u_data, rt_data = \
        read_data(directory + basename)

    assert num_particles == n_particles

    # Note that we have the nodes stored in the output, not the elements.
    assert x_data.shape == (n_radius + 1, )
    assert u_data.shape == (len(t), n_particles, 2, n_radius + 1)


def test_NonlinearProblemBlock():

    num_points = 16

    mesh = dfx.mesh.create_unit_square(comm, num_points, num_points)

    V = dfx.fem.functionspace(mesh, ("Lagrange", 1))
    W = scifem.create_real_functionspace(mesh, value_shape=())

    # %% manufactured solution
    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

    n = ufl.FacetNormal(mesh)

    g = ufl.dot(ufl.grad(u_ex), n)
    f = ufl.div(ufl.grad(u_ex))
    h = scifem.assemble_scalar(u_ex * ufl.dx)

    # %% Problem setup
    u = dfx.fem.Function(V)
    v = ufl.TestFunction(V)

    u_mean = dfx.fem.Function(W)
    v_mean = ufl.TestFunction(W)

    pde = -ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + g * v * ufl.ds - f * v * ufl.dx
    pde += u_mean * v * ufl.dx
    constraint = (u - h) * v_mean * ufl.dx

    F = [pde, constraint]

    # %% Solver setup
    w = [u, u_mean]

    problem = NonlinearProblemBlock(F, w)

    solver = NewtonSolver(comm, problem)

    its, success = solver.solve(w)

    error_form = dfx.fem.form((u - u_ex)**2 * ufl.dx)

    error = scifem.assemble_scalar(error_form)

    # Initial tests gave an error of 4.4e-5 for 16 points.
    assert np.isclose(error, 0., atol=5e-5)
