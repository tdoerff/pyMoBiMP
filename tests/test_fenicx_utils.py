from mpi4py.MPI import COMM_WORLD as comm

import os

from pyMoBiMP.fenicsx_utils import read_data


def test_read_data_DFN_shape():

    n_particles = 4
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
