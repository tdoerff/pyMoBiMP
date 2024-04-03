import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm_world

import pytest

import ufl

from pyMoBiMP.cahn_hilliard_utils import cahn_hilliard_form


def test_instantiate_two_particle_form():

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    n_particle = 2

    # The single-component element
    elem_c = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2, n_particle)
    elem_mu = elem_c

    mixed_element = elem_c * elem_mu

    V = dfx.fem.FunctionSpace(
        mesh, mixed_element
    )  # A mixed two-component function space

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 1.0)

    F = cahn_hilliard_form(u, u0, dt)
