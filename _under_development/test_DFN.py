import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import pytest

from CH_4_DFN_grid import (TestCurrent,
                           create_1p1_DFN_mesh,
                           create_particle_summation_measure,
                           DFN_function_space,
                           physical_setup,
                           Voltage)


def test_physical_setup():

    mesh = create_1p1_DFN_mesh(comm)
    V = DFN_function_space(mesh)

    dA_R = create_particle_summation_measure(mesh)
    _, a_ratios, L, Ls = physical_setup(V)

    assert np.isclose(dfx.fem.assemble_scalar(dfx.fem.form(a_ratios * dA_R)),
                      1.0)

    assert np.isclose(
        dfx.fem.assemble_scalar(dfx.fem.form(Ls * a_ratios * dA_R)),
        L)


@pytest.mark.parametrize("I_global_value", [0., 1e-3, 0.1, 1.0, 10., 100.])
def test_Voltage_constant_mu(I_global_value: float):

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)

    # Test for mu = 0
    u = dfx.fem.Function(V)  # mu = 0

    I_global = dfx.fem.Constant(mesh, I_global_value)

    voltage = Voltage(u, I_global)

    print(f"L * V(I={I_global.value}, mu=0) = ", voltage.L * voltage.value)

    assert np.isclose(voltage.L * voltage.value, -I_global.value)

    # Test for mu = 1
    _, mu = u.split()
    mu.x.array[:] = 1.

    assert np.isclose(voltage.L * voltage.value, -I_global.value - voltage.L)


def test_Voltage_constant_I_global():

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)

    u = dfx.fem.Function(V)

    I_global = dfx.fem.Constant(mesh, 1.)

    voltage = Voltage(u, I_global)

    # Test for mu = 1
    _, mu = u.split()

    for mu_value in [-100., -10., -1., 0., 1e-3, 0.1, 1.0, 10., 100.]:

        mu.x.array[:] = mu_value

        assert np.isclose(voltage.L * voltage.value,
                          -I_global.value - mu_value * voltage.L)


def test_particle_current():

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)

    u = dfx.fem.Function(V)

    I_global = dfx.fem.Constant(mesh, 0.)

    voltage = Voltage(u, I_global)

    test_current = TestCurrent(u, voltage, I_global)

    for I_global_value in [0., 1e-3, 0.1, 1.0, 10., 100.]:
        I_global.value = I_global_value

        voltage.update()

        I_global_ref = test_current.compute_current()

        print(I_global_value, I_global_ref)

        assert np.isclose(I_global_value, I_global_ref)
