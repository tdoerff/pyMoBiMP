"""create_multi_particle_animation.py

Usage: python create_multi_particle_animation.py <simulation_base_name>
"""

import argparse
import dolfinx as dfx
import importlib
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import os
import scipy as sp
import tqdm
import ufl

from pyMoBiMP.fenicsx_utils import read_data
from pyMoBiMP.gmsh_utils import dfx_spherical_mesh
from pyMoBiMP.plotting_utils import PyvistaAnimation


def f_of_q_default(q):
    a = 1.5
    b = 0.2
    cc = 5.

    return np.log(q / (1 - q)) + a * (1 - 2 * q) + b * np.pi * cc * np.cos(np.pi * cc * q)


def identity_function(n_points):
    """Creates a identity dfx.fem.Function on a unit intervall"""

    # Create a linspace array.
    mesh = dfx.mesh.create_unit_interval(comm, n_points)
    V = dfx.fem.FunctionSpace(mesh, ("CG", 1))

    c = dfx.fem.Function(V)
    c.interpolate(lambda x: x[0])

    return c


def get_chemical_potential(experiment):
    """
    Extract the chemical potential function from the given experiment script.
    """

    experiment = importlib.import_module(args.experiment)
    free_energy = experiment.Simulation.free_energy

    # Get a function get an identity function [0,1] -> [0,1]
    c = identity_function(128)
    V = c.function_space

    # Hold the function values for later use as coordinate while
    # interpolation.
    c_val = c.x.array[:]

    # Compute chemical potential as derivative of free energy
    # w.r.t. c.
    c = ufl.variable(c)
    chem_pot_expr = dfx.fem.Expression(
        ufl.diff(free_energy(c, ufl.ln, ufl.sin), c),
        V.element.interpolation_points())

    # Map the expression to a dfx.fem.Function.
    chem_pot = dfx.fem.Function(V)
    chem_pot.interpolate(
        chem_pot_expr)

    # Retrieve the function values.
    f_val = chem_pot.x.array[:]

    # Construct the interpolation polynomial.
    f_of_q = sp.interpolate.interp1d(c_val, f_val)

    return f_of_q


# The follwowing two helper functions are taken from:
# https://stackoverflow.com/a/52014520
def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip()  # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])

        if value == "False":
            value = False
        elif value == "True":
            value = True

        print(key, value)

    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def write_output(args, anim):
    output = args.output

    print(args.additional_args)

    additional_args = parse_vars(args.additional_args)

    # Write as a movie file.
    if output[-4:] == ".mp4" or \
        output[-4:] == ".mpg" or \
            output[-5:] == ".mpeg":
        anim.get_mp4_animation(output, additional_options=additional_args)

    # Write as a GIF file.
    elif output[-4:] == ".gif":
        anim.get_gif_animation(output, additional_options=additional_args)

    else:
        raise ValueError(f"Format not recognized ({output})!")


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
    parser.add_argument("--cmap", type=str, default="graphite")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--clipped", action="store_true")
    parser.add_argument("-e", "--experiment", type=str)
    parser.add_argument("--additional-args",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as strings.")

    args = parser.parse_args()

    # Read the data
    # -------------

    # First read the XMDF simulation output
    filebasename = args.filename

    return_container = read_data(filebasename)

    num_particles, t, x_data, u_data, rt_data = return_container

    if len(x_data.shape) == 3:
        x_data = x_data[0:1, 0, :]

    # read chemical potential from experiment script
    if args.experiment is not None:
        try:
            f_of_q = get_chemical_potential(args.experiment)

        except ImportError as e:

            print(e)
            print(
                f"import `{args.experiment}` not found. Fall back to default.")

            # Use default chemical potential
            f_of_q = f_of_q_default
    else:
        f_of_q = None

    # Create or get the mesh(es)
    # --------------------------

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
        clipped=args.clipped,
        specular=1.0,
        metallic=0.5,
    )

    write_output(args, anim)
