# %%
import dolfinx as dfx

import h5py

from dolfinx.fem.petsc import NonlinearProblem

from matplotlib import pyplot as plt
# plt.style.use('fivethirtyeight')

from mpi4py import MPI

import numpy as np

from pathlib import Path

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    cahn_hilliard_form,
    charge_discharge_stop,
    AnalyzeOCP,
    y_of_c,
    c_of_y,
    populate_initial_data)

from pyMoBiMP.fenicsx_utils import (evaluation_points_and_cells,
                           get_mesh_spacing,
                           time_stepping,
                           NewtonSolver,
                           FileOutput as FileOutputBase,
                           Fenicx1DOutput)

from pyMoBiMP.gmsh_utils import dfx_spherical_mesh

from pyMoBiMP.plotting_utils import (
    add_arrow,
    plot_charging_cycle,
    plot_time_sequence,
    PyvistaAnimation,
    animate_time_series)

comm_world = MPI.COMM_WORLD

# %%
# Discretization
# --------------

# Set up the mesh
n_elem = 128

mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

dx_cell = get_mesh_spacing(mesh)

print(f"Cell spacing: h = {dx_cell}")

# For later plotting use
x = np.linspace(0, 1, 101)
points_on_proc, cells = evaluation_points_and_cells(mesh, x)

# Initial timestep size
dt = dfx.fem.Constant(mesh, dx_cell * 0.01)

# %%
elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

elem_c = elem1
elem_mu = elem1

num_particles = 2

multi_particle_element = ufl.MixedElement(
    [[elem_c, ] * num_particles,
        [elem_mu, ] * num_particles]
    )

V = dfx.fem.FunctionSpace(mesh, multi_particle_element)

# %%
# The mixed-element functions
u = dfx.fem.Function(V)
u0 = dfx.fem.Function(V)

# %%
# Compute the chemical potential df/dc
a = 6. / 4
b = 0.2
cc = 5

free_energy = lambda u, log, sin: u * log(u) + (1-u) * log(1-u) + a * u * (1 - u) + b * sin(cc * np.pi * u)

eps = 1e-4


# %%
# Experimental setup
# ------------------

# charging current
I_charge = dfx.fem.Constant(mesh, 1e-1)

T_final = 2. / I_charge.value  # ending time

def experiment(t, u, I_charge, **kwargs):

    return charge_discharge_stop(t, u, I_charge, c_of_y = c_of_y, **kwargs)

event_params = dict(I_charge=I_charge, stop_on_full=False, stop_at_empty=False, cycling=False, logging=True)

# %%
# The variational form
# --------------------

form_weights = None
M = lambda c: c * (1 - c)
theta = 1.
lam = 0.1
grad_c_bc = lambda c: 0.

alpha = 100.

h = ufl.Circumradius(mesh)

y, mu = ufl.split(u)
y0, mu0 = ufl.split(u0)

y1, y2 = ufl.split(y)
y01, y02 = ufl.split(y0)

mu1, mu2 = ufl.split(mu)
mu01, mu02 = ufl.split(mu0)

y1, y2 = ufl.variable(y1), ufl.variable(y2)
c1, c2 = c_of_y(y1), c_of_y(y2)

dcdy1, dcdy2 = ufl.diff(c1, y1), ufl.diff(c2, y2)

v_c, v_mu = ufl.TestFunctions(V)

v_c1, v_c2 = ufl.split(v_c)
v_mu1, v_mu2 = ufl.split(v_mu)

c1, c2 = ufl.variable(c1), ufl.variable(c2)
f_e1, f_e2 = free_energy(c1, ufl.ln, ufl.sin), free_energy(c2, ufl.ln, ufl.sin)

mu_chem1, mu_chem2 = ufl.diff(f_e1, c1), ufl.diff(f_e2, c2)

mu_theta1, mu_theta2 = theta * mu1 + (theta - 1.0) * mu01, theta * mu2 + (theta - 1.0) * mu02

r = ufl.SpatialCoordinate(mesh)

# adaptation of the volume element due to geometry
if form_weights is not None:
    s_V = form_weights["volume"]
    s_A = form_weights["surface"]
else:
    s_V = 4 * np.pi * r**2
    s_A = 2 * np.pi * r**2

dx = ufl.dx  # The volume element
ds = ufl.ds  # The surface element

flux1, flux2 = M(c1) * ufl.grad(mu_theta1), M(c2) * ufl.grad(mu_theta2)

n = n_1 = n_2 = ufl.FacetNormal(mesh)

# I * (A_1 + A_2) = I_1 * A_1 + I_2 * A_2
I_charge1 = 2 * I_charge - ufl.dot(n_2, flux2)
I_charge2 = 2 * I_charge - ufl.dot(n_1, flux1)

F1 = cahn_hilliard_form(mesh, (y1, mu1), (y01, mu01), (v_c1, v_mu1), dt,
                        M=M, c_of_y=c_of_y,
                        free_energy=lambda c: free_energy(c, ufl.ln, ufl.sin),
                        theta=1., lam=0.1,
                        I_charge=I_charge1)

F1 += ufl.dot(ufl.grad(v_mu1), n_1) * (mu1 - mu2) * s_A * ds
F1 -= alpha / h * (mu1 - mu2) * v_mu1 * s_A * ds

F2 = cahn_hilliard_form(mesh, (y2, mu2), (y02, mu02), (v_c2, v_mu2), dt,
                        M=M, c_of_y=c_of_y,
                        free_energy=lambda c: free_energy(c, ufl.ln, ufl.sin),
                        theta=1., lam=0.1,
                        I_charge=I_charge2)

F2 += ufl.dot(ufl.grad(v_mu2), n_2) * (mu2 - mu1) * s_A * ds
F2 -= alpha / h * (mu2 - mu1) * v_mu2 * s_A * ds

F = F1 + F2

# %%
# boundary conditions
# -------------------

def boundary_locator(x):
    return np.isclose(x[0], 1)

# facets
tdim = mesh.topology.dim - 1

facets = dfx.mesh.locate_entities_boundary(mesh, tdim, boundary_locator)
dofs1 = dfx.fem.locate_dofs_topological((V.sub(1).sub(0), V.sub(1).sub(1)), tdim, facets)
dofs2 = dfx.fem.locate_dofs_topological((V.sub(1).sub(1), V.sub(1).sub(0)), tdim, facets)

bcs = [
    dfx.fem.dirichletbc(u.sub(1).sub(1), dofs1, V.sub(1).sub(0)),
    dfx.fem.dirichletbc(u.sub(1).sub(0), dofs2, V.sub(1).sub(1))
]

# %%
# Initial data
# ------------

u_ini = dfx.fem.Function(V)

# Constant
c_ini_fun = lambda x: eps * np.ones_like(x[0])


# Store concentration-like quantity into state vector
# ---------------------------------------------------

V_c, _ = V.sub(0).collapse()

c_ini = dfx.fem.Function(V_c)

c_ini.sub(0).interpolate(lambda x: 1. * c_ini_fun(x))
c_ini.sub(1).interpolate(lambda x: 1. * c_ini_fun(x))

y_ini1 = dfx.fem.Expression(y_of_c(c_ini.sub(0)),
                            c_ini.sub(0).function_space.element.interpolation_points())
y_ini2 = dfx.fem.Expression(y_of_c(c_ini.sub(1)),
                            c_ini.sub(1).function_space.element.interpolation_points())

u_ini.sub(0).sub(0).interpolate(y_ini1)
u_ini.sub(0).sub(1).interpolate(y_ini2)

# Store chemical potential into state vector
# ------------------------------------------

c_ini1 = ufl.variable(c_ini.sub(0))
dFdc1 = ufl.diff(free_energy(c_ini1, ufl.ln, ufl.sin), c_ini1)

W = u_ini.sub(1).sub(0).function_space
u_ini.sub(1).sub(0).interpolate(dfx.fem.Expression(dFdc1, W.element.interpolation_points()))

c_ini2 = ufl.variable(c_ini.sub(1))
dFdc2 = ufl.diff(free_energy(c_ini2, ufl.ln, ufl.sin), c_ini2)

W = u_ini.sub(1).sub(1).function_space
u_ini.sub(1).sub(1).interpolate(dfx.fem.Expression(dFdc2, W.element.interpolation_points()))

u_ini.x.scatter_forward()

# %%
problem = NonlinearProblem(F, u)

solver = NewtonSolver(comm_world, problem)

# %%
class Output(Fenicx1DOutput):

    def extract_output(self, u_state, t):

        V = self.u_state.function_space

        num_comp = V.num_sub_spaces

        output_snapshot = []

        for i_comp in range(V.num_sub_spaces):

            V_sub = V.sub(i_comp)

            for k_comp in range(V_sub.num_sub_spaces):

                values = u_state.sub(i_comp).sub(k_comp).eval(self.x_eval, self.cells)

                output_snapshot.append(values)

        return output_snapshot


class FileOutput(FileOutputBase):

    def extract_output(self, u_state, t):
        V = self.u_state.function_space

        num_vars = V.num_sub_spaces

        ret = []

        for i in range(num_vars):

            V_sub, _ = V.sub(i).collapse()

            if i == 0:
                name = "y"
            elif i == 1:
                name = "mu"
            else:
                raise ValueError(f"No component with index {i} available!")

            num_comp = V_sub.num_sub_spaces

            for j in range(num_comp):

                func = self.u_state.sub(i).sub(j)
                func.name = name + f"_{j}"

                ret.append(func)

        return ret

# %%
# Set up experiment
# -----------------

u.interpolate(u_ini)

n_out = 501

output_np = Output(u, np.linspace(0, T_final, n_out), x)

results_folder = Path("simulation_output")
results_folder.mkdir(exist_ok=True, parents=True)

filename = results_folder / "CH_4_min_2_particle.xdmf"

output_xdmf = FileOutput(u, np.linspace(0, T_final, 51), filename=filename)

rt_analysis = AnalyzeOCP(c_of_y=c_of_y, filename=results_folder / "CH_4_min_1D_rt.txt")

# %%
# Run the experiment
# ------------------

time_stepping(
    solver,
    u,
    u0,
    T_final,
    dt,
    dt_max=1e-3,
    dt_min=1e-12,
    tol=1e-7,
    event_handler=lambda *args, **kwargs: False,
    output=(output_np, output_xdmf),
    runtime_analysis=None,
    **event_params,
)
