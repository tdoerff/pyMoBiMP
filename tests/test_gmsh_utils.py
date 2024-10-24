from mpi4py.MPI import COMM_WORLD as comm

from pyMoBiMP.gmsh_utils import dfx_spherical_mesh


def test_instantiate_shperical_mesh():

    dfx_spherical_mesh(comm)
