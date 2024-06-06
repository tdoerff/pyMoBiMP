import numpy as np

import pytest

from CH_4_min_n_particles import compute_particle_current_densities


@pytest.mark.parametrize("num_particles", [1, 2, 5])
def test_current(num_particles):

    mus = 3 * (2 * np.random.random(num_particles) - 1)
    As = 4 * np.pi * np.random.random(num_particles)
    Ls = 1e3 * (2 * np.random.random(num_particles) - 1)
    i_charge = 1.0

    i_charges = compute_particle_current_densities(
        mus, As, Ls, i_charge
    )

    A = sum(As)

    aas = As / A

    # Check if the sum over the individual particle currents matches
    # the total current.

    I_charge_sum = sum([a_ * i_ for a_, i_ in zip(aas, i_charges)])

    assert np.isclose(i_charge, I_charge_sum)
