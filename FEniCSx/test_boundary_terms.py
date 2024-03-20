# %% [markdown]
# # Use ```ufl``` form to compute boundary term
#
# We have a function defined on $[0,1] \ni r$ and want to obtain the value at $r=1$.

# %%
import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm_world

import numpy as np

import ufl

from fenicsx_utils import evaluation_points_and_cells


# %%
def test_boundary_terms():
    n_elem = 128

    mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    V = dfx.fem.FunctionSpace(mesh, elem1)

    # %%
    u = dfx.fem.Function(V)

    r = ufl.SpatialCoordinate(mesh)

    u.interpolate(dfx.fem.Expression(0.5 * r**2, V.element.interpolation_points()))
    u.interpolate(lambda r: np.sin(r[0]))

    # %%
    u_bc = dfx.fem.form(r**2 * u * ufl.ds)
    u_bc_form = dfx.fem.assemble_scalar(u_bc)

    print(u_bc_form)

    # %%
    x, cell = evaluation_points_and_cells(mesh, np.array([1.]))

    u_bc_eval = u.eval(x, cell)[0]

    print(u_bc_eval)

    # %%
    assert np.isclose(u_bc_eval, u_bc_form)
