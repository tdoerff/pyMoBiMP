from typing import Literal
import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm_world

import pytest

import ufl

from pyMoBiMP.cahn_hilliard_utils import c_of_y


@pytest.mark.parametrize('num_particles', [1, 2, 3])
def test_instantiate_c_of_y(num_particles):

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    # The single-component element
    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    elem_c = elem1
    elem_mu = elem1

    multi_particle_element = ufl.MixedElement(
        [[elem_c, ] * num_particles,
         [elem_mu, ] * num_particles]
    )

    V = dfx.fem.FunctionSpace(mesh, multi_particle_element)

    u = dfx.fem.Function(V)

    y, _ = ufl.split(u)

    [c_of_y(yi) for yi in ufl.split(y)]
