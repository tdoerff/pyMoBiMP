# %%
# %load_ext autoreload
# %autoreload 2

# %%
import dolfinx as dfx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from mpi4py import MPI

import numpy as np

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    AnalyzeOCP,
    cahn_hilliard_form,
    charge_discharge_stop,
    c_of_y,
    _free_energy as free_energy,
    populate_initial_data,
    y_of_c,
)

from pyMoBiMP.fenicsx_utils import (
    get_mesh_spacing,
    time_stepping,
    FileOutput,
    # NewtonSolver,
    # NonlinearProblem
)

from pyMoBiMP.gmsh_utils import dfx_spherical_mesh

comm_world = MPI.COMM_WORLD


def log(*args, **kwargs):
    if comm_world.rank == 0:
        print(*args, **kwargs)


# %%
# create the mesh

mesh, ct, ft = dfx_spherical_mesh(comm_world, resolution=0.5)

# %%
# Discretization details
# ----------------------

# spatial
dx_cell = get_mesh_spacing(mesh)

log(f"Cell spacing: h = {dx_cell}")

# temporal
dt = dfx.fem.Constant(mesh, dx_cell * 0.01)

log(f"timestep size: dt = {dt.value:1.3e}")

# %%
elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

mixed_element = elem1 * elem1

V = dfx.fem.FunctionSpace(mesh, mixed_element)  # A mixed two-component function space

# %%
# The mixed-element functions
u = dfx.fem.Function(V)
u0 = dfx.fem.Function(V)

# %%
# Experimental setup
# ------------------

T_final = 0.2  # ending time

# charging current
I_charge = dfx.fem.Constant(mesh, 0.01)

coords = ufl.SpatialCoordinate(mesh)
r2 = ufl.dot(coords, coords)

y, mu = u.split()
c = c_of_y(y)

c_bc_form = dfx.fem.form(r2 * c * ufl.ds)


def experiment(t, u, I_charge, cell_voltage, **kwargs):

    return charge_discharge_stop(t, u, I_charge, c_bc_form, c_of_y=c_of_y)


event_params = dict(I_charge=I_charge, stop_at_empty=False, cycling=False)

# %%
# The variational form
# --------------------
params = dict(I_charge=I_charge)

form_weights = dict(surface=1.0, volume=1.0)

F = cahn_hilliard_form(
    u,
    u0,
    dt,
    free_energy=free_energy,
    theta=0.75,
    c_of_y=c_of_y,
    M=lambda c: 1.0 * c * (1 - c),
    lam=0.1,
    form_weights=form_weights,
    **params,
)

# %%
# Initial data
# ------------

log(">>> Setting up initial data ...")

u_ini = dfx.fem.Function(V)

# Constant
eps = 1e-3
def c_ini_fun(x): return eps * np.ones_like(x[0])


populate_initial_data(u_ini, c_ini_fun, free_energy, y_of_c=y_of_c)

# %%

log(">>> Setting up the solver ...")

problem = NonlinearProblem(F, u)

solver = NewtonSolver(comm_world, problem)

# %%

log(">>> Running the simulation ...")

u.interpolate(u_ini)

n_out = 51

output = FileOutput(
    u, np.linspace(0, T_final, 51), filename="simulation_output/CH_4_true_3d.xdmf"
)

rt_analysis = AnalyzeOCP(u, c_of_y=c_of_y)

time_stepping(
    solver,
    u,
    u0,
    T_final,
    dt,
    dt_increase=1.01,
    dt_max=1e-2,
    event_handler=experiment,
    output=output,
    runtime_analysis=rt_analysis,
    **event_params,
)
