from mpi4py import MPI as MPI
import numpy as np
import ufl
import dolfinx as dfx
import pyvista as pv

n_rad = 64
n_part = 12

# %% Create elements and nodes on first proc
# ==========================================
if MPI.COMM_WORLD.rank == 0:

    radial_grid = np.linspace(0, 1, n_rad)
    particle_grid = np.linspace(0, 1, n_part)

    rr, pp = np.meshgrid(radial_grid, particle_grid)

    coords_grid = np.stack((rr, pp)).transpose((-1, 1, 0)).copy()

    coords_grid_flat = coords_grid.reshape(-1, 2).copy()

    # All the radial connections
    elements_radial = [[
        [n_part * i + k, n_part * (i + 1) + k] for i in range(n_rad - 1)]
                        for k in range(n_part)]

    elements_radial = np.array(elements_radial).reshape(-1, 2)

    elements = elements_radial

else:
    coords_grid_flat = np.empty((0, 2), dtype=np.float64)
    elements = np.empty((0, 2), dtype=np.int64)

# %% Create the global mesh on all procs
# ======================================

gdim = 2
shape = "interval"
degree = 1

cell = ufl.Cell(shape, geometric_dimension=gdim)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

try:
    mesh = dfx.mesh.create_mesh(
        MPI.COMM_WORLD, elements[:, :2], coords_grid_flat, domain)
except Exception as e:
    print(e)

# %% Show the grid
# ================

V = dfx.fem.FunctionSpace(mesh, ("CG", 1))

u = dfx.fem.Function(V)
u.interpolate(lambda x: x[0] + x[1])


def plot_solution_on_grid(u):

    V = u.function_space

    topology, cell_types, x = dfx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    grid['u'] = u.x.array

    plotter = pv.Plotter()

    plotter.add_mesh(grid, show_edges=True, show_vertices=True)
    plotter.add_axes()
    plotter.add_bounding_box()

    plotter.show()


plot_solution_on_grid(u)
