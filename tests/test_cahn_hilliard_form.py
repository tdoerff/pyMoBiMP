import basix

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
    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)

    elem_c = elem1
    elem_mu = elem1

    particle_element = basix.ufl.mixed_element([elem_c, elem_mu])

    V = dfx.fem.functionspace(mesh, particle_element)

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    dt = dfx.fem.Constant(mesh, 1.0)

    F = cahn_hilliard_form(u, u0, dt)


@pytest.mark.parametrize("num_particles", [1, 2, 3])
def test_instantiate_multi_particle_particle_form(num_particles):

    # Set up the mesh
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    # The single-component element
    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)

    elem_c = elem1
    elem_mu = elem1

    multi_particle_element = basix.ufl.mixed_element(
        [basix.ufl.mixed_element([elem_c, ] * num_particles),
            basix.ufl.mixed_element([elem_mu, ] * num_particles)]
    )

    V = dfx.fem.functionspace(
        mesh, multi_particle_element
    )  # A mixed two-component function space

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    v = ufl.TestFunction(V)

    dt = dfx.fem.Constant(mesh, 1.0)

    Fs = []

    for y, mu, y0, mu0, vc, vmu in zip(
        *ufl.split(u), *ufl.split(u0), *ufl.split(v)
    ):
        Fs.append(cahn_hilliard_form(mesh, (y, mu), (y0, mu0), (vc, vmu), dt))

    F = sum(Fs)
