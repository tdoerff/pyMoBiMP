# file plotting_utils.py

import ipywidgets
from matplotlib import pyplot as plt
import numpy as np


def add_arrow(line, position=None, direction='right', size=15, color=None):
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
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def plot_charging_cycle(q, mu_bc, eps):
    fig, ax = plt.subplots()

    line_mu, = ax.plot(q, -mu_bc, label=r"$\left. \mu \right|_{\partial \omega_I}$")
    # line_f, = ax.plot(q, -f_bar, label=r"$\overline{f(c)}$")

    # TODO: use AD or something else to generalize
    q_plot = np.linspace(eps, 1-eps, 101)
    dFdc = np.log(q_plot / (1 - q_plot))

    ax.plot(q_plot, -dFdc, 'r--', label=r"$f(q)$")

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

    return fig, ax


def animate_time_series(output, c_of_y):

    fig, ax = plt.subplots()

    x, t_out, data_out = output.get_output(return_time=True, return_coords=True)

    data_out = np.array(data_out).squeeze()

    it_max = len(data_out)

    def update(it = 10):

        c = c_of_y(data_out[it][0])

        line.set_ydata(c)
        fig.canvas.draw_idle()

    line, _, _ = ax.plot(x, c_of_y(data_out[0][0]))

    ax.set_ybound(0, 1)

    ipywidgets.interact(update, it=ipywidgets.IntSlider(min=0, max=it_max - 1, step=1, value=0))