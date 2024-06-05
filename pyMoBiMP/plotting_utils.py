# file plotting_utils.py

from collections.abc import Callable

import dolfinx

import ipywidgets

import math

from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt

import os

import vtk  # noqa: 401 necessary to use latex labels in pyvista
import pyvista

import scipy as sp

from typing import List, Optional, Tuple

from .fenicsx_utils import Fenicx1DOutput


def add_arrow(line, position=None, direction="right", size=15, color=None):
    """
    add an arrow to a line.

    Copied from
    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot (2024/02/28)

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


def plot_time_sequence(output, c_of_y, plot_mu=True):

    if plot_mu:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax = plt.subplots()

    # get the data
    # ------------

    if isinstance(output, Fenicx1DOutput):
        r, t_out, data_out = output.get_output(
            return_time=True, return_coords=True)

    elif isinstance(output, List) or isinstance(output, Tuple):
        # Unpack the list
        r, t_out, data_out = output

    else:
        raise TypeError("`output` is not List or Tuple or Fenicsx1DOutput!")

    r = np.array(r)[:, 0]

    data_out = np.array(data_out)

    for it_out, (data_t, t) in enumerate(zip(data_out, t_out)):

        y_t = data_t[0]
        mu_t = data_t[1]

        c_t = c_of_y(y_t)

        if plot_mu:
            ax = axs[0]

        color = (t / t_out[-1], 0, 0)

        ax.plot(r, c_t, color=color)

        ax.set_ylabel("$c$")

        if plot_mu:
            ax = axs[1]

            color = (0, 0, t / t_out[-1])

            ax.plot(r, mu_t, color=color)

            ax.set_ylabel(r"$\mu$")

    ax.set_xlabel("$r$")

    if not plot_mu:
        axs = ax

    return fig, axs


def animate_time_series(output, c_of_y):

    fig, ax = plt.subplots()

    x, t_out, data_out = output.get_output(return_time=True, return_coords=True)

    r = np.array(x)[:, 0]

    data_out = np.array(data_out).squeeze()

    it_max = len(data_out)

    def update(it=10):

        c = c_of_y(data_out[it][0])

        line.set_ydata(c)
        fig.canvas.draw_idle()

    line, = ax.plot(r, c_of_y(data_out[0][0]))

    ax.set_ybound(0, 1)

    ipywidgets.interact(
        update, it=ipywidgets.IntSlider(min=0, max=it_max - 1, step=1, value=0)
    )


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

    # Note that this construction ensures that Nx * Ny > num_particles.
    Nx = math.floor(N)
    Ny = math.ceil(N)

    # Anyhow, make sure that we have enough grid positions for all particles.
    # Round-off errors might cause situations where N**2 = num_particles - eps
    # s.t. there might be one row missing in the resulting grid.
    # E.g.: sqrt(9) ~ 2.99999 => Nx = 2, Ny = 3 => Nx * Ny < num_particles.
    if Nx * Ny < num_particles:
        Nx += 1

    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    plot_grid = np.array([x.flatten(), y.flatten()]).T

    return plot_grid


class PyvistaAnimation:

    def __init__(
        self,
        output: Fenicx1DOutput | List[npt.NDArray] | Tuple[npt.NDArray],
        rt_data: Optional[npt.NDArray] = None,
        plot_c_of_r: bool = True,
        c_of_y: Callable[[npt.ArrayLike], npt.ArrayLike] = lambda y: y,
        f_of_q: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None,
        meshes: Optional[dolfinx.mesh.Mesh] = None,
        specular: float = 1.0,
        auto_close=False,
        interactive_update=True,
        clipped: Optional[bool] = None,
        **plotter_kwargs
    ):

        # Sanitize input
        # --------------
        plotter_kwargs.update(specular=specular)
        self.plotter_kwargs = plotter_kwargs

        # get the data
        # ------------

        if isinstance(output, Fenicx1DOutput):
            r, t_out, data_out = output.get_output(
                return_time=True, return_coords=True)

        elif isinstance(output, List) or isinstance(output, Tuple):
            # Unpack the list
            r, t_out, data_out = output

        else:
            raise TypeError("Input is not List or Tuple or Fenicsx1DOutput!")

        self.r = np.array(r)[:, 0]

        # make sure even for a single particle we can use the upcoming code
        shape = len(t_out), -1, 2, len(self.r)
        self.data_out = np.array(data_out).reshape(shape)

        # Convert y to c.
        self.data_out[..., 0, :] = c_of_y(self.data_out[..., 0, :])

        self.t_out = np.array(t_out)

        self.meshes = meshes

        # Initialize pyvista plotter
        # ==========================

        # figure out how many subplots we need.
        num_plots = 1  # At least one plot

        self.plot_cell_voltage = rt_data is not None

        if self.plot_cell_voltage:
            num_plots += 1

        if plot_c_of_r:
            num_plots += 1
        self.plot_c_of_r = plot_c_of_r

        self.num_plots = num_plots

        if num_plots == 1:
            self.plotter = pyvista.Plotter()
        elif num_plots == 2:
            self.plotter = pyvista.Plotter(shape="1/1")
        elif num_plots == 3:
            self.plotter = pyvista.Plotter(shape="1/2")
        else:
            raise RuntimeError(f"num_plots = {num_plots} is invalid.")

        # The clipped sphere plot
        # -----------------------
        if num_plots > 1:
            self.plotter.subplot(self.num_plots - 1)

        # The contruct the (clipped) grids.
        if clipped is None:
            num_particles = shape[1]

            clipped = (num_particles == 1)

        self.grids = self.set_up_pv_grids(clipped)

        # TODO: optionally clip grids
        self.time_label = self.plotter.add_text(f"t = {self.t_out[0]:1.3f}")

        # radial charge distribution
        # --------------------------
        if self.plot_c_of_r:
            self.plotter.subplot(0)

            chart = pyvista.Chart2D()

            c = self.data_out[0, ..., 0, :]

            chart.line(r, c, color=(0, 0, 0))

            chart.y_range = [0, 1]

            chart.x_label = r"$r$"
            chart.y_label = r"$c$"

            self.plotter.add_chart(chart)

            # Initialized actors
            self.c_t_chart = chart
            self.c_t_lines = [chart.line(self.r, c_i, 'r') for c_i in c]

        # cell voltage plot
        # -----------------
        if self.plot_cell_voltage:
            self.plotter.subplot(int(plot_c_of_r))

            chart = pyvista.Chart2D(x_label=r"$q$", y_label=r"$\mu$")

            q = rt_data[:, 1]
            mu = rt_data[:, -1]

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

            self.plotter.add_chart(chart)

            # Initialized actors
            self.q_mu_chart = chart
            self.q_mu_scatter = chart.scatter([], [])

        self.show(auto_close=auto_close, interactive_update=interactive_update)

    def set_up_pv_grids(self,
                        clipped: bool,
                        shift: float = 2.):

        grids = []

        num_particles = len(self.meshes)

        plot_grid = assemble_plot_grid(num_particles)

        for i_particle in range(num_particles):

            V = dolfinx.fem.FunctionSpace(self.meshes[i_particle], ("CG", 1))

            topology, cell_types, x = dolfinx.plot.vtk_mesh(V)

            # center point of the current grid.
            x0 = shift * plot_grid[i_particle]

            # Shift grid to center
            x[:, :2] += x0

            grid = pyvista.UnstructuredGrid(topology, cell_types, x)

            if clipped:
                grid = grid.clip_box([x0[0], x0[0]+1, x0[1], x0[1]+1, 0, 0+1])

            self.update_on_grid(i_particle, 0, V.mesh, grid)

            grids.append(grid)

            self.plotter.add_mesh(grid, **self.plotter_kwargs)

        return grids

    def it_max_and_update(self):
        """Returns it_max and update to be used, e.g., in ipywidget"""

        return len(self.data_out), self.update

    def get_slider_widget(self):
        """Sets up and returns an ipywidget slider object."""

        it_max, update = self.it_max_and_update()

        it_slider = ipywidgets.IntSlider(min=0, max=it_max - 1, step=1, value=0)

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

    def get_mp4_animation(
            self, filename: str | os.PathLike = "anim.mp4"):
        """Stores a gif file."""

        it_max, update = self.it_max_and_update()

        self.plotter.open_movie(str(filename))

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

    def show(self, **kwargs):
        self.plotter.show(**kwargs)

    def update(self, it):

        self._update_data_on_all_grids(it)

        self._update_time_label(it)

        if self.plot_c_of_r:
            self._update_c_t(it)

        if self.plot_cell_voltage:
            self._update_q_mu(it)

    def _update_time_label(self, it):

        # FIXME: adjust in case of less than three plots
        if self.num_plots > 1:
            self.plotter.subplot(self.num_plots - 1)

        for actor, content in self.plotter.actors.items():
            if content == self.time_label:
                content.SetVisibility(False)

        self.time_label = self.plotter.add_text(f"t = {self.t_out[it]:1.3f}")

    def update_on_grid(self, i_particle, i_t, dfx_mesh, pv_grid):

        x = pv_grid.cast_to_pointset().points

        r_grid = np.sqrt((x**2).sum(axis=-1))

        c = self.data_out[i_t, i_particle, 0, :]

        u_3d = np.interp(r_grid, self.r, c)

        pv_grid["u"] = u_3d

    def _update_data_on_all_grids(self, it):

        for i_particle, (grid, dfx_mesh) in enumerate(zip(self.grids, self.meshes)):

            self.update_on_grid(i_particle, it, dfx_mesh, grid)

    def _update_clipped_grid(self):

        clipped = self.grid.clip_box([0., 1., 0., 1., 0., 1.], crinkle=False)

        # This should do the trick to update the clipped
        # data without re-plotting whole grid.
        self.plotter.mesh["u"] = clipped["u"]

        self.plotter.update()

    def _update_c_t(self, it):

        c = self.data_out[it, ..., 0, :]

        [self.c_t_chart.remove_plot(c_t_line) for c_t_line in self.c_t_lines]

        self.c_t_lines = [self.c_t_chart.line(self.r, c_i, 'r') for c_i in c]

    def _update_q_mu(self, it):

        # The list of all radial charge distributions at each time.
        cs = self.data_out[it, ..., 0, :].reshape(-1, len(self.r))

        num_particles = self.data_out.shape[1]

        # Compute charge ratio to maximum charge.
        q = np.array([sp.integrate.trapezoid(
            3 * self.r**2 * c, self.r, axis=-1) for c in cs]).sum(axis=0) / num_particles

        # chemical potential time series at the outer edge for each particle.
        mu = self.data_out[it, ..., 1, -1]

        self.q_mu_chart.remove_plot(self.q_mu_scatter)

        self.q_mu_scatter = self.q_mu_chart.scatter([q], [-mu], color='r')


def plot_charging_cycle(I_q_mu_bcs, f_A, eps=1e-3):

    # [ ] adjust plot size
    # [ ] adjust spacing around x=0 and x=1

    # chart = pyvista.Chart2D()

    # q_plot = np.linspace(eps, 1-eps, 101)
    # f = f_A(q_plot)

    # chart.line(q_plot, -f, color="tab:orange", style="--", label=r"f_A")

    # for i, (I, q, mu) in enumerate(I_q_mu_bcs):

    #     color = (0, 0, 0.2 + 0.79 * i / (max(len(I_q_mu_bcs), 2) - 1))

    #     chart.line(q, -mu, color=color, label=rf"I = {I:1.3e}")

    # return chart

    fig, ax = plt.subplots()

    q_plot = np.linspace(eps, 1-eps, 101)
    f = f_A(q_plot)

    ax.plot(q_plot, -f, "r--", label=r"$f_A$")

    for i, (I, q, mu) in enumerate(I_q_mu_bcs):

        color = (0, 0, 0.2 + 0.79 * i / (max(len(I_q_mu_bcs), 2) - 1))

        if isinstance(I, float):
            label = rf"I = {I:1.1e}"
        elif isinstance(I, str):
            label = I
        else:
            label = None

        (line, ) = ax.plot(q, -mu, color=color, label=label, alpha=0.5)

        add_arrow(line)

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\mu$")

    ax.legend()

    return ax
