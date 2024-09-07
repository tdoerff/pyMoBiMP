import basix
import dolfinx as dfx
from dolfinx.fem.petsc import NonlinearProblem as NonlinearProblemBase
from dolfinx.nls.petsc import NewtonSolver

from mpi4py.MPI import COMM_WORLD as comm, SUM

import numpy as np

from petsc4py import PETSc

import pyvista as pv

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    compute_chemical_potential,
    _free_energy as free_energy)

from pyMoBiMP.fenicsx_utils import (
    FileOutput,
    RuntimeAnalysisBase,
    time_stepping)

# %% Helper functions
# ===================


def plot_solution_on_grid(u):

    V = u.function_space

    topology, cell_types, x = dfx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    grid['u'] = u.x.array

    plotter = pv.Plotter()

    warped = grid.warp_by_scalar('u')

    plotter.add_mesh(warped, show_edges=True, show_vertices=False, )
    plotter.add_axes()

    plotter.show()


class AnalyzeOCP(RuntimeAnalysisBase):
    def setup(self, u_state, c_of_y, I_global, Ls, a_ratios, *args, **kwargs):
        super().setup(u_state, *args, **kwargs)

        # Function space(s) and mesh information
        V = self.u_state.function_space
        mesh = V.mesh

        V0, _ = V.sub(0).collapse()

        coords = ufl.SpatialCoordinate(mesh)
        r = coords[0]

        # Create integral measure on the particle surface
        fdim = mesh.topology.dim - 1

        facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

        facet_markers = np.full_like(facets, 1)

        facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

        dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
        dA_R = dA(1)

        # By integrating over the boundary we get a measure of the
        # number of particles.
        num_particles = dfx.fem.assemble_scalar(
            dfx.fem.form(dfx.fem.Constant(mesh, 1.) * dA_R)
        )
        num_particles = mesh.comm.allreduce(num_particles, op=SUM)

        # Weighted mean affinity parameter taken from particle surfaces.
        L_ufl = a_ratios * Ls * dA_R
        L = dfx.fem.assemble_scalar(dfx.fem.form(L_ufl))  # weighted reaction affinity
        L = mesh.comm.allreduce(L, op=SUM)

        y, mu = self.u_state.split()

        # compute state of charge
        c = c_of_y(y)

        self.soc_form = dfx.fem.form(3 / num_particles * c * r**2 * ufl.dx)

        # Compute cell potential
        OCP = - Ls / L * a_ratios * mu * dA_R

        self.V_cell_form = dfx.fem.form(- (I_global / L) * dA_R + OCP)

    def analyze(self, t):

        mesh = self.u_state.function_space.mesh

        soc = dfx.fem.assemble_scalar(self.soc_form)
        soc = mesh.comm.allreduce(soc, op=SUM)

        V_cell = dfx.fem.assemble_scalar(self.V_cell_form)
        V_cell = mesh.comm.allreduce(V_cell, op=SUM)

        self.data.append([soc, V_cell])

        return super().analyze(t)


# %% Grid setup
# =============
n_rad = 16
n_part = 192

# Nodes
# -----
radial_grid = np.linspace(0, 1, n_rad)
particle_grid = np.linspace(0, 1, n_part)

rr, pp = np.meshgrid(radial_grid, particle_grid)

coords_grid = np.stack((rr, pp)).transpose((-1, 1, 0)).copy()

if comm.rank == 0:
    coords_grid_flat = coords_grid.reshape(-1, 2).copy()
else:
    coords_grid_flat = np.empty((0, 2), dtype=np.float64)

# Elements
# --------
# All the radial connections
elements_radial = [
    [[n_part * i + k, n_part * (i + 1) + k] for i in range(n_rad - 1)]
    for k in range(n_part)
]

elements_radial = np.array(elements_radial).reshape(-1, 2)

# Connections between particles
elements_bc = (n_rad - 1) * n_part + np.array([[k, k + 1] for k in range(n_part - 1)])
elements_bc = []  # With elements at the outer edge the integration fails.

if comm.rank == 0:
    elements = np.array(list(elements_bc) + list(elements_radial))
else:
    elements = np.empty((0, 2), dtype=np.int64)

# %% The DOLFINx grid
# -------------------

gdim = 2
shape = "interval"
degree = 1

domain = ufl.Mesh(
    basix.ufl.element("Lagrange",
                      shape,
                      degree,
                      shape=(coords_grid_flat.shape[1],)))

mesh = dfx.mesh.create_mesh(comm, elements[:, :2], coords_grid_flat, domain)

# %% The DOLFINx function space
# -----------------------------
elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
V = dfx.fem.functionspace(mesh, basix.ufl.mixed_element([elem1, elem1]))

V0, _ = V.sub(0).collapse()  # <- auxiliary space for coefficient functions

# %% Create integral measure on the particle surface
# --------------------------------------------------
fdim = mesh.topology.dim - 1

facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

facet_markers = np.full_like(facets, 1)

facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
dA_R = dA(1)

# %% Physical setup of the problem
# ================================

# particle parameters
Rs = dfx.fem.Function(V0)
Rs.x.array[:] = 1.
Rs.x.scatter_forward()

As = 4 * np.pi * Rs**2

# Reaction affinity of the particles.
L_mean = 10.
L_var_rel = 0.1

Ls = dfx.fem.Function(V0)
Ls.x.array[:] = L_mean + \
    L_var_rel * (2 * np.random.random(Ls.x.array.shape) - 1)

A_ufl = As * dA_R
A = dfx.fem.assemble_scalar(dfx.fem.form(A_ufl))
A = mesh.comm.allreduce(A, op=SUM)

a_ratios = As / A

# Weighted mean reaction affinity parameter taken from particle surfaces.
L_ufl = a_ratios * Ls * dA_R
L = dfx.fem.assemble_scalar(dfx.fem.form(L_ufl))
L = mesh.comm.allreduce(L, op=SUM)

# %% The FEM form
# ===============

u = dfx.fem.Function(V)
u0 = dfx.fem.Function(V)

y, mu = ufl.split(u)
y0, mu0 = ufl.split(u0)

v_c, v_mu = ufl.TestFunctions(V)

I_global = dfx.fem.Constant(mesh, 1e-1)

OCP = - Ls / L * a_ratios * mu * dA_R

V_cell_form = dfx.fem.form(OCP - I_global / L * a_ratios * dA_R)

V_cell = dfx.fem.Constant(mesh, 0.0)

I_particle = - Ls * (mu + V_cell)


def callback():

    V_cell_value = dfx.fem.assemble_scalar(V_cell_form)
    V_cell.value = comm.allreduce(V_cell_value, op=SUM)


# With this dirty hack we make sure the voltage is updated
# per iteration within the Newton solver
class NonlinearProblem(NonlinearProblemBase):
    def form(self, x):
        callback()
        super().form(x)


theta = 1.0
dt = dfx.fem.Constant(mesh, 1e-6)

c = c_of_y(y)

V0, dofs = V.sub(0).collapse()
r = dfx.fem.Function(V0)
r.interpolate(lambda x: x[0])


def M(c):
    return c * (1 - c)


lam = 0.1


def grad_c_bc(c):
    return 0.


s_V = 4 * np.pi * r**2
s_A = 2 * np.pi * r**2

dx = ufl.dx  # The volume element

mu_chem = compute_chemical_potential(free_energy, c)
mu_theta = theta * mu + (theta - 1.0) * mu0

flux = M(c) * mu_theta.dx(0)

F1 = s_V * (c_of_y(y) - c_of_y(y0)) * v_mu * dx
F1 += s_V * flux * v_mu.dx(0) * dt * dx
F1 -= I_particle * s_A * v_mu * dt * dA_R

F2 = s_V * mu * v_c * dx
F2 -= s_V * mu_chem * v_c * dx
F2 -= lam * (s_V * c.dx(0) * v_c.dx(0) * dx)
F2 += grad_c_bc(c) * (s_A * v_c * dA_R)

F = F1 + F2

residual = dfx.fem.form(F)


# %% DOLFINx problem and solver setup
# ===================================

problem = NonlinearProblem(F, u)
solver = NewtonSolver(comm, problem)
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# %% Initial data
# ===============
u0.sub(0).x.array[:] = -6  # This corresponds to roughly c = 1e-3


if __name__ == "__main__":

    dt_min = 1e-9
    dt_max = 1e-3

    dt.value = 1e-8

    T_final = 65.0
    tol = 1e-4

    I_global.value = 0.01

    u.x.scatter_forward()
    u0.x.scatter_forward()

    iterations, success = solver.solve(u)
    print(iterations, success)

    # %% Runtime analysis and output
    # ==============================
    rt_analysis = AnalyzeOCP(u,
                             c_of_y,
                             I_global,
                             Ls,
                             a_ratios,
                             filename="CH_4_DFN_rt.txt")

    output = FileOutput(u,
                        np.linspace(0, T_final, 101),
                        filename="CH_4_DFN.xdmf",
                        variable_transform=c_of_y)

    # %% Run the simulation
    # =====================
    time_stepping(
        solver,
        u,
        u0,
        T_final,
        dt,
        dt_max=dt_max,
        dt_min=dt_min,
        dt_increase=1.1,
        tol=tol,
        runtime_analysis=rt_analysis,
        output=output
    )
