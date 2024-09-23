import basix

from typing import Literal
import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm_world

import pytest

import ufl

from pyMoBiMP.cahn_hilliard_utils import c_of_y

from pyMoBiMP.fenicsx_utils import strip_off_xdmf_file_ending


@pytest.mark.parametrize('num_particles', [1, 2, 3])
def test_instantiate_c_of_y(num_particles):

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

    V = dfx.fem.functionspace(mesh, multi_particle_element)

    u = dfx.fem.Function(V)

    y, _ = ufl.split(u)

    [c_of_y(yi) for yi in ufl.split(y)]


def test_strip_off_ending():

    file_name_base = "foo"

    xdmf_file_name = file_name_base + ".xdmf"
    h5_file_name = file_name_base + ".h5"

    bar_file_name = file_name_base + ".bar"

    assert strip_off_xdmf_file_ending(xdmf_file_name) == file_name_base
    assert strip_off_xdmf_file_ending(h5_file_name) == file_name_base
    assert strip_off_xdmf_file_ending(bar_file_name, '.bar') == file_name_base
