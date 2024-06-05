"""create_multi_particle_animation.py

Usage: python create_multi_particle_animation.py <simulation_base_name>
"""

import argparse
import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import os
import tqdm

from pyMoBiMP.fenicsx_utils import read_data
from pyMoBiMP.gmsh_utils import dfx_spherical_mesh
from pyMoBiMP.plotting_utils import PyvistaAnimation


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
    parser.add_argument("-o", "--output", type=str,
                        default="multi_particle_anim.mp4")
    parser.add_argument("-c", "--clim", type=float, nargs=2, default=[0., 1.])
    parser.add_argument("--cmap", type=str, default="fire")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--clipped", action="store_true")

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

    pbar = tqdm.tqdm(len(t))

    class MyPyvistaAnimation(PyvistaAnimation):

        def update(self, i):

            super().update(i)
            pbar.update(1)

    anim = MyPyvistaAnimation(
        (x_data, t, u_data),
        rt_data,
        c_of_y=lambda y: np.exp(y) / (1 + np.exp(y)),
        f_of_q=f_of_q,
        meshes=meshes,
        cmap=args.cmap,
        clim=args.clim,
        auto_close=args.close,
        clipped=args.clipped
    )

    anim.get_mp4_animation(args.output)
