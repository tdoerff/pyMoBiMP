import numpy as np

from pyMoBiMP.fenicsx_utils import read_data
from pyMoBiMP.plotting_utils import PyvistaAnimation


def test_PyvistaAnimation():

    num_particles, t, x_data, u_data, rt_data = \
        read_data("tests/data/DFN_Simulation/output")

    anim = PyvistaAnimation(
        (x_data, t, u_data),
        rt_data,
        c_of_y=lambda y: np.exp(y) / (1 + np.exp(y)),
        auto_close=True,
        clipped=True,
    )

    anim.get_gif_animation("anim.gif")
    anim.get_mp4_animation("anim.mp4")
