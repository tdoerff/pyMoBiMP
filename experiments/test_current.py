import numpy as np

import pytest

from CH_4_min_n_particles import compute_particle_current_densities


def test_test():

    num_particles = 2

    mus = 3 * (2 * np.random.random(num_particles) - 1)
    As = 4 * np.pi * np.ones(num_particles)
    Ls = 1e3 * (2 * np.random.random(num_particles) - 1)
    I_charge = 1.0

    i_charges = compute_particle_current_densities(
        mus, As, Ls, I_charge
    )

    A = sum(As)

    aas = As / A

    assert np.isclose(I_charge,
                      sum([a_ * i_ for a_, i_ in zip(aas, i_charges)]))
