"""create_multi_particle_animation.py

Usage: python create_multi_particle_animation.py <simulation_base_name>
"""

import argparse
import dolfinx as dfx
import h5py
import math
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import os
import pyvista as pv
import shutil
import tqdm

from pyMoBiMP.gmsh_utils import dfx_spherical_mesh


# open issues
# [ ] add r-c plot
# [ ] add q-V plot
# [ ] make it a class

def assemble_plot_grid(num_particles: int):
    """Find an arrangement for closest to a square for a given number of particles.

    Parameters
    ----------
    num_particles : int
        number of particles to distribute

    Returns
    -------
    plot_grid : np.array
        Index mapping returning the plot coordinates for a given particle index.
    """

    N = num_particles**0.5

    Nx = math.floor(N)
    Ny = math.ceil(N)

    if Nx * Ny < N:
        Nx += 1

    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    plot_grid = np.array([x.flatten(), y.flatten()]).T

    return plot_grid


def set_up_pv_grids(meshes: list[dfx.mesh.Mesh],
                    plotter: pv.Plotter,
                    i_t0: int = 0,
                    **plotter_kwargs):

    grids = []

    num_particles = len(meshes)

    plot_grid = assemble_plot_grid(num_particles)

    for i_particle in range(num_particles):

        V = dfx.fem.FunctionSpace(meshes[i_particle], ("CG", 1))

        topology, cell_types, x = dfx.plot.vtk_mesh(V)

        x[:, :2] += 2. * plot_grid[i_particle]

        grid = pv.UnstructuredGrid(topology, cell_types, x)

        update_on_grid(i_particle, i_t0, V.mesh, grid)

        grids.append(grid)

        plotter.add_mesh(grid, **plotter_kwargs)

    return grids


def update_on_grid(i_particle, i_t, dfx_mesh, pv_grid):

    r_sphere = np.sqrt((dfx_mesh.geometry.x[:, :]**2).sum(axis=-1))

    # TODO: make this replaceable.
    def c_of_y(y): return np.exp(y) / (1 + np.exp(y))

    c = c_of_y(u_data[i_t, i_particle, 0, :])
    r = x_data[:, 0]

    u_3d = np.interp(r_sphere, r, c)

    pv_grid["u"] = u_3d


class SimulationFile(h5py.File):
    """A file handler class to open pyMoBiMP simulation output.

    The main purpose of the class is to wrap a copy operation to the
    standard h5py file handler to avoid deadlocks when opening files
    from a running simulation.

    Attributes
    ----------
    _file_name : str
        file name pointing to the simulation output
    _file_name_tmp : str
        the temporary file name of the copied file
    """

    def __init__(self, file_name_base):
        """Construct file name and temporary file name.

        Parameters
        ----------
        file_name_base : str | pathlib.Path
            file name pointing to the file name base or XDMF or H5 file.
        """

        # Strip off the file ending for uniform file handling
        if file_name_base[-3:] == ".h5":
            file_name_base[-3:]

        elif file_name_base[-5:] == ".xdmf":
            file_name_base[-5:]

        # Make sure we have to absolute file name at hand.
        file_name_base = os.path.abspath(file_name_base)

        self._file_name = file_name_base + ".h5"
        self._file_name_tmp = file_name_base + "_tmp" + ".h5"

        # To avoid file locks, copy the current version of the file to a tmp file.
        shutil.copy(self._file_name, self._file_name_tmp)

        # Advise base class to open the tmp file.
        super().__init__(self._file_name_tmp)

    def __exit__(self, *args, **kwargs):
        """
        Ensures that after the file operation is done, the temporary file is deleted.
        """

        super().__exit__(self, *args, **kwargs)

        # ... and remove it
        os.remove(self._file_name_tmp)


def read_data(filebasename):

    print(f"Read data from {filebasename} ...")

    with SimulationFile(filebasename) as f:
        print(f["Function"].keys())

        num_particles = len(f["Function"].keys()) // 2

        print(f"Found {num_particles} particles.")

        # grid coordinates
        x_data = f["Mesh/mesh/geometry"][()]

        t_keys = f["Function/y_0"].keys()

        # time steps (convert from string to float)
        t = [float(t.replace("_", ".")) for t in t_keys]

        # list of data stored as numpy arrays
        u_data = np.array([
            [(f[f"Function/y_{i_part}"][u_key][()].squeeze(),
              f[f"Function/mu_{i_part}"][u_key][()].squeeze())
             for i_part in range(num_particles)] for u_key in t_keys])

    # It is necessary to sort the input by the time.
    sorted_indx = np.argsort(t)

    t = np.array(t)[sorted_indx]
    u_data = np.array(u_data)[sorted_indx]

    # Read the runtime analysis output.
    rt_data = np.loadtxt(filebasename + "_rt.txt")

    return num_particles, t, x_data, u_data, rt_data


def f_of_q(q):
    a = 1.5
    b = 0.2
    cc = 5.

    return np.log(q / (1 - q)) + a * (1 - 2 * q) + b * np.pi * cc * np.cos(np.pi * cc * q)


if __name__ == "__main__":

    # Parse input arguments
    # ---------------------
    filename = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        prog=filename,
        description="Creates an animation of multi-particle simulation.",
        )

    parser.add_argument("filename", type=str)
    parser.add_argument("-m", "--mesh-file", type=str)
    parser.add_argument("-r", "--mesh-resolution", type=float, default=1.0)
    parser.add_argument("-o", "--output", type=str, default="multi_particle_anim.mpeg")
    parser.add_argument("-c", "--clim", type=float, nargs=2, default=[0., 1.])
    parser.add_argument("--cmap", type=str, default="fire")
    parser.add_argument("--close", action="store_true")

    args = parser.parse_args()

    # Read the data
    # -------------

    # First read the XMDF simulation output
    filebasename = args.filename

    num_particles, t, x_data, u_data, rt_data = read_data(filebasename)

    # Create or get the mesh(es)
    # --------------------------
    # TODO: add the option for individual particle meshes.

    # Create or read-in a single particle mesh
    if args.mesh_file is None or os.path.isfile(args.mesh_file):
        mesh_3d, _, _ = \
            dfx_spherical_mesh(comm, resolution=args.mesh_resolution, optimize=False)

        if args.mesh_file is not None:
            print(f">>> Write mesh to {args.mesh_file}")
            with dfx.io.XDMFFile(comm, args.mesh_file, "w") as file:
                file.write_mesh(mesh_3d)

    else:
        print(f">>> Read mesh from {args.mesh_file}")
        with dfx.io.XDMFFile(comm, args.mesh_file, "r") as file:
            mesh_3d = file.read_mesh()

    meshes = [mesh_3d, ] * num_particles
    # Create visualization
    # --------------------

    plotter = pv.Plotter(shape="1/1")

    plotter.subplot(1)
    grids = set_up_pv_grids(meshes, plotter, clim=args.clim, cmap=args.cmap)

    plotter.show(auto_close=args.close, interactive_update=True)

    plotter.subplot(0)

    chart = pv.Chart2D()

    q = rt_data[:, 1]
    mu = rt_data[:, 3]

    chart.line(q, -mu, color="k", label=r"$\mu\vert_{\partial\omega_I}$")

    chart.x_range = [0, 1]
    eps = 0.5
    chart.y_range = [min(mu) - eps, max(mu) + eps]

    if f_of_q is not None:
        eps = 1e-3
        q = np.linspace(eps, 1 - eps, 101)

        chart.line(q, -f_of_q(q), color="tab:orange", style="--", label=r"$f_A$")

        eps = 0.5
        chart.y_range = \
            [min(chart.y_range[0], min(-f_of_q(q) - eps)),
                max(chart.y_range[1], max(-f_of_q(q) + eps))]

    plotter.add_chart(chart)

    # Write to file
    # -------------

    plotter.open_movie(args.output)

    def update_all_grid(i_t, grids):
        for i_particle in range(num_particles):
            update_on_grid(i_particle, i_t, meshes[i_particle], grids[i_particle])

    def update(i):
        update_all_grid(i, grids)
        plotter.update()

    for it in tqdm.trange(len(t)):

        update(it)

        plotter.write_frame()
