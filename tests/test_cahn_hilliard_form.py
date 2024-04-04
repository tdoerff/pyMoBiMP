import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm_world

import pytest

import ufl

from pyMoBiMP.cahn_hilliard_utils import cahn_hilliard_form


def test_instantiate_single_particle_form():

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    # The single-component element
    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    elem_c = elem1
    elem_mu = elem1

    mixed_element = elem_c * elem_mu

    V = dfx.fem.FunctionSpace(mesh, mixed_element)  # A mixed two-component function space

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 1.0)

    F = cahn_hilliard_form(u, u0, dt)


def test_instantiate_two_ingle_particle_form():

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    # The single-component element
    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    elem_c = elem1
    elem_mu = elem1

    mixed_element = elem_c * elem_mu

    two_particle_element = mixed_element * mixed_element

    V = dfx.fem.FunctionSpace(mesh, two_particle_element)  # A mixed two-component function space

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 1.0)

    F = cahn_hilliard_form(u, u0, dt)
