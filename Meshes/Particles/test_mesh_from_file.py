import dolfinx

from mpi4py import MPI

import numpy as np

import ufl

if __name__ == "__main__":

    vertices = np.loadtxt("Meshes/Particles/Spherical/singleParticle_Tom1_vol.node")
    cells = np.loadtxt("Meshes/Particles/Spherical/singleParticle_Tom1_vol.elem", dtype=np.int32)

    gdim = 3
    shape = "tetrahedron"
    degree = 1

    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells[:, :4] - 1, vertices, domain)
