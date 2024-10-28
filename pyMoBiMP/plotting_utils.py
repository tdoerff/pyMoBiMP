# MIT License
#
# Copyright (c) 2024 Tom Doerffel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# See LICENSE file.

import colorcet as cc

from collections.abc import Callable

import dolfinx

import ipywidgets

from matplotlib import colors as clrs
from matplotlib import pyplot as plt

from mpi4py.MPI import COMM_WORLD as comm

import numpy as np
import numpy.typing as npt

import os

import vtk  # noqa: 401 necessary to use latex labels in pyvista
import pyvista

from typing import List, Optional, Tuple

from .fenicsx_utils import Fenicx1DOutput
from .gmsh_utils import dfx_spherical_mesh


_fire = cc.cm.CET_L3
graphite_colormap = clrs.ListedColormap(
    _fire(np.linspace(0.15, 0.95, 101)), name='graphite')


def plot_solution_on_grid(u: dolfinx.fem.Function):

    V = u.function_space

    topology, cell_types, x = dolfinx.plot.vtk_mesh(V)

    n_particles = np.max(x)
    x[:, 1] /= n_particles

    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    grid['u'] = u.x.array

    plotter = pyvista.Plotter()

    warped = grid.warp_by_scalar('u')

    plotter.add_mesh(warped, show_edges=True, show_vertices=False, show_scalar_bar=True)
    plotter.add_axes()
    plotter.add_bounding_box()

    plotter.show()


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


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


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

    primes = prime_factors(num_particles)

    primes_0 = primes[::2]
    primes_1 = primes[1::2]

    Nx = np.prod(primes_0)
    Ny = np.prod(primes_1)

    if Ny > 2 * Nx:
        Ny //= primes[0]
        Nx *= primes[0]

    elif Nx > 2 * Ny:
        Nx //= primes[0]
        Ny *= primes[0]

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

    # FIXME: We don't we see phase separation within the particles?

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

        if "cmap" not in self.plotter_kwargs.keys():
            self.plotter_kwargs.update(cmap=graphite_colormap)

        elif self.plotter_kwargs["cmap"] == "graphite":
            self.plotter_kwargs.update(cmap=graphite_colormap)

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

        self.r = np.array(r).squeeze()

        # make sure even for a single particle we can use the upcoming code
        shape = len(t_out), -1, 2, len(self.r)
        self.data_out = np.array(data_out).reshape(shape)
        self.rt_data = rt_data

        # Convert y to c.
        self.data_out[..., 0, :] = c_of_y(self.data_out[..., 0, :])

        self.t_out = np.array(t_out)

        if meshes is not None:
            self.meshes = meshes
        else:
            num_particles = self.data_out.shape[1]
            mesh_3d, _, _ = dfx_spherical_mesh(comm, resolution=0.5, optimize=False)
            self.meshes = [mesh_3d, ] * num_particles

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

            chart = pyvista.Chart2D(x_label=r"$SoC$", y_label=r"$V$")

            # Retrieve the charge.
            q = rt_data[:, 1]

            # Retrieve voltage.
            V = rt_data[:, -1]

            # Retrieve time.
            t = rt_data[:, 0]

            dt = np.diff(t)

            dq = np.zeros_like(q)
            dq[:-1] = np.diff(q) / dt
            dq[-1] = dq[-2]

            eps = 1e-7

            chart.line(q[dq > eps],
                       V[dq > eps],
                       color="r",
                       label=r"$V_{cell}(I>0)$")
            chart.line(q[dq < 0],
                       V[dq < 0],
                       color="b",
                       label=r"$V_{cell}(I<0)$")

            chart.x_range = [0, 1]
            eps = 0.5
            chart.y_range = [min(V) - eps, max(V) + eps]

            if f_of_q is not None:
                eps = 1e-3
                q = np.linspace(eps, 1 - eps, 101)

                chart.line(q, -f_of_q(q),
                           color="tab:orange", style="--", label=r"$f_A$")

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

            V = dolfinx.fem.functionspace(self.meshes[i_particle], ("CG", 1))

            topology, cell_types, x = dolfinx.plot.vtk_mesh(V)

            # center point of the current grid.
            x0 = shift * plot_grid[i_particle]

            # Shift grid to center
            x[:, :2] += x0

            grid = pyvista.UnstructuredGrid(topology, cell_types, x)

            if clipped:
                grid = grid.clip_box([x0[0], x0[0]+1, x0[1], x0[1]+1, 0, 0+1])

            self.update_on_grid(i_particle, 0, grid)

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
            self, filename: str | os.PathLike = "anim.gif",
            additional_options: dict = dict()):
        """Stores a gif file."""

        it_max, update = self.it_max_and_update()

        self.plotter.open_gif(str(filename), **additional_options)

        for it in range(it_max):

            update(it)

            self.plotter.write_frame()

        self.plotter.close()

    def get_mp4_animation(
            self, filename: str | os.PathLike = "anim.mp4",
            additional_options: dict = dict()):
        """Stores a gif file."""

        it_max, update = self.it_max_and_update()

        self.plotter.open_movie(str(filename), **additional_options)

        for it in range(it_max):

            update(it)

            self.plotter.write_frame()

        self.plotter.close()

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

        if self.num_plots > 1:
            self.plotter.subplot(self.num_plots - 1)

        for actor, content in self.plotter.actors.items():
            if content == self.time_label:
                content.SetVisibility(False)

        self.time_label = self.plotter.add_text(f"t = {self.t_out[it]:1.3f}")

    def update_on_grid(self, i_particle, i_t, pv_grid):

        x = pv_grid.cast_to_pointset().points

        x0 = x.mean(axis=0)
        x -= x0

        r_grid = np.sqrt((x**2).sum(axis=-1))

        c = self.data_out[i_t, i_particle, 0, :]

        u_3d = np.interp(r_grid, self.r, c)

        pv_grid["u"] = u_3d

    def _update_data_on_all_grids(self, it):

        for i_particle, grid in enumerate(self.grids):

            self.update_on_grid(i_particle, it, grid)

        self.plotter.update()

    def _update_c_t(self, it):

        c = self.data_out[it, ..., 0, :]

        [self.c_t_chart.remove_plot(c_t_line) for c_t_line in self.c_t_lines]

        self.c_t_lines = [self.c_t_chart.line(self.r, c_i, 'r') for c_i in c]

    def _update_q_mu(self, it):

        t = self.t_out[it]

        q = np.interp(t, self.rt_data[:, 0], self.rt_data[:, 1])
        V = np.interp(t, self.rt_data[:, 0], self.rt_data[:, -1])

        self.q_mu_chart.remove_plot(self.q_mu_scatter)

        self.q_mu_scatter = self.q_mu_chart.scatter([q], [V], color='r')
