# MIT License
#
# Copyright (c) 2024 Tom Doerffel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# See LICENSE file.

"""
dfn_ultis.py: This files contains functionality specific to the
Dyle-Fuller-Newman-like implementation of the battery model.
"""

import basix

import dolfinx as dfx

from mpi4py import MPI

import numpy as np

import scifem

from typing import List

import ufl


def DFN_function_space(mesh: dfx.mesh.Mesh) -> List[dfx.fem.FunctionSpace]:
    # %% The DOLFINx function space
    # -----------------------------
    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    V = dfx.fem.functionspace(mesh, basix.ufl.mixed_element([elem1, elem1]))

    W = scifem.create_real_functionspace(mesh)

    return V, W


def create_1p1_DFN_mesh(comm: MPI.Intracomm,
                        n_rad: int = 16,
                        n_part: int = 192) -> dfx.mesh.Mesh:

    if comm.rank == 0:
        radial_grid = np.linspace(0, 1, n_rad)
        particle_grid = np.linspace(0, 1, n_part)

        rr, pp = np.meshgrid(radial_grid, particle_grid)

        coords_grid = np.stack((rr, pp)).transpose((-1, 1, 0)).copy()

        coords_grid.shape

        coords_grid_flat = coords_grid.reshape(-1, 2).copy()

        # All the radial connections
        elements_radial = [
            [[n_part * i + k, n_part * (i + 1) + k] for i in range(n_rad - 1)]
            for k in range(n_part)
        ]

        elements_radial = np.array(elements_radial).reshape(-1, 2)

        # Connections between particles
        elements_bc = (n_rad - 1) * n_part + np.array(
            [[k, k + 1] for k in range(n_part - 1)]
        )
        elements_bc = []  # With elements at the outer edge the integration fails.

        elements = np.array(list(elements_bc) + list(elements_radial))

    else:
        coords_grid_flat = np.empty((0, 2), dtype=np.float64)
        elements = np.empty((0, 2), dtype=np.int64)

    gdim = 2
    shape = "interval"
    degree = 1

    domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))

    mesh = dfx.mesh.create_mesh(comm, elements[:, :gdim], coords_grid_flat, domain)
    return mesh


def create_particle_summation_measure(mesh: dfx.mesh.Mesh) -> ufl.Measure:

    fdim = mesh.topology.dim - 1

    facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

    facet_markers = np.full_like(facets, 1)

    facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
    dA_R = dA(1)

    return dA_R
