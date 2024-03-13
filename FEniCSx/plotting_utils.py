# file plotting_utils.py

from collections.abc import Callable

import dolfinx

import ipywidgets

from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt

import os

import pyvista

import scipy as sp

from typing import Optional

from fenicsx_utils import Fenicx1DOutput
from gmsh_utils import dfx_spherical_mesh


def add_arrow(line, position=None, direction="right", size=15, color=None):
    """
    add an arrow to a line.

    Copied from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot (2024/02/28)

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size,
    )


def plot_charging_cycle(q, mu_bc, eps):
    fig, ax = plt.subplots()

    (line_mu,) = ax.plot(q, -mu_bc, label=r"$\left. \mu \right|_{\partial \omega_I}$")
    # line_f, = ax.plot(q, -f_bar, label=r"$\overline{f(c)}$")

    ax.set_xlabel(r"q")

    ax.legend()

    add_arrow(line_mu, position=0.4002)

    return fig, ax


def plot_time_sequence(output, c_of_y):

    fig, axs = plt.subplots(2, 1, sharex=True)

    x, t_out, data_out = output.get_output(return_time=True, return_coords=True)

    data_out = np.array(data_out).squeeze()

    for it_out, (data_t, t) in enumerate(zip(data_out, t_out)):

        y_t = data_t[0]
        mu_t = data_t[1]

        c_t = c_of_y(y_t)

        ax = axs[0]

        color = (it_out / len(t_out), 0, 0)

        ax.plot(x, c_t, color=color)

        ax = axs[1]

        color = (0, 0, it_out / len(t_out))

        ax.plot(x, mu_t, color=color)

    return fig, axs


def animate_time_series(output, c_of_y):

    fig, ax = plt.subplots()

    x, t_out, data_out = output.get_output(return_time=True, return_coords=True)

    data_out = np.array(data_out).squeeze()

    it_max = len(data_out)

    def update(it=10):

        c = c_of_y(data_out[it][0])

        line.set_ydata(c)
        fig.canvas.draw_idle()

    line, _, _ = ax.plot(x, c_of_y(data_out[0][0]))

    ax.set_ybound(0, 1)

    ipywidgets.interact(
        update, it=ipywidgets.IntSlider(min=0, max=it_max - 1, step=1, value=0)
    )


class PyvistaAnimation:

    def __init__(
        self,
        output: Fenicx1DOutput,
        c_of_y: Callable[[npt.ArrayLike], npt.ArrayLike] = lambda y: y,
        mesh_3d: Optional[dolfinx.mesh.Mesh] = None,
        res: float = 1.0,
        specular: float = 1.0,
        **plotter_kwargs
    ):

        # Sanitize input
        # --------------
        plotter_kwargs.update(specular=specular)
        self.plotter_kwargs = plotter_kwargs

        # get the data
        # ------------
        r, t_out, data_out = output.get_output(return_time=True, return_coords=True)
        self.r = np.array(r)[:, 0]

        self.data_out = np.array(data_out).squeeze()
        self.data_out[:, 0, :] = c_of_y(self.data_out[:, 0, :])

        self.t_out = np.array(t_out)

        if mesh_3d is None:
            mesh_3d, _, _ = dfx_spherical_mesh(resolution=res)

        # Create the function space and Function object for holding the data
        # ------------------------------------------------------------------
        V_3d = dolfinx.fem.functionspace(mesh_3d, ("CG", 1))

        self.u_3d = dolfinx.fem.Function(V_3d)

        # Initialize pyvista plotter
        # --------------------------

        # The unclipped grid.
        self.grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_3d))

        self._update_data_on_grid(0)

        grid_clipped = self.grid.clip_box([0., 1., 0., 1., 0., 1.], crinkle=False)

        self.plotter = pyvista.Plotter()

        self.plotter.add_mesh(grid_clipped, **plotter_kwargs)

        self.time_label = self.plotter.add_text(f"t = {self.t_out[0]:1.3f}")

        self.show()

    def it_max_and_update(self):
        """Returns it_max and update to be used, e.g., in ipywidget"""

        return len(self.data_out), self.update

    def get_slider_widget(self):
        """Sets up and returns an ipywidget slider object."""

        it_max, update = self.it_max_and_update()

        it_slider=ipywidgets.IntSlider(min=0, max=it_max - 1, step=1, value=0)

        widget = ipywidgets.interact(update, it=it_slider)

        return widget

    def get_gif_animation(
        self, filename: str | os.PathLike = "anim.gif"):
        """Stores a gif file."""

        it_max, update = self.it_max_and_update()

        self.plotter.open_gif(str(filename))

        for it in range(it_max):

            update(it)

            self.plotter.write_frame()

        self.plotter.close()

    def write_vtk_output(
        self, filename: str | os.PathLike = "output.bp"):

        u = self.u_3d
        u.name = "c"
        mesh = u.function_space.mesh
        comm = mesh.comm

        # writer = dolfinx.io.VTXWriter(comm, filename, u)
        file = dolfinx.io.VTKFile(comm, filename, "w")

        file.write_mesh(mesh)

        it_max, _ = self.it_max_and_update()

        for it in range(it_max):

            self._update_data_on_grid(it)

            file.write_function(u, self.t_out[it])

        file.close()

    def show(self):
        self.plotter.show()

    def update(self, it):

        self._update_data_on_grid(it)

        self._update_clipped_grid()

        self._update_time_label(it)

    def _update_time_label(self, it):

        for actor, content in self.plotter.actors.items():
            if content == self.time_label:
                content.SetVisibility(False)

        self.time_label = self.plotter.add_text(f"t = {self.t_out[it]:1.3f}")

    def _update_data_on_grid(self, it):

        c = self.data_out[it, 0, :]

        poly = sp.interpolate.interp1d(self.r, c, fill_value="extrapolate")

        self.u_3d.interpolate(lambda x: poly((x[0]**2 + x[1]**2 + x[2]**2)**0.5))

        self.grid["u"] = self.u_3d.x.array

    def _update_clipped_grid(self):

        clipped = self.grid.clip_box([0., 1., 0., 1., 0., 1.], crinkle=False)

        # This should do the trick to update the clipped
        # data without re-plotting whole grid.
        self.plotter.mesh["u"] = clipped["u"]

        self.plotter.update()
