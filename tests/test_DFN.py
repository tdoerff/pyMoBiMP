import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np
import pytest
import scifem
import ufl

from pyMoBiMP.dfn_battery_model import (
    TestCurrent,
    create_1p1_DFN_mesh,
    create_particle_summation_measure,
    DFN_function_space,
    DefaultPhysicalSetup as PhysicalSetup,
    voltage_form
)


def test_physical_setup():

    mesh = create_1p1_DFN_mesh(comm)
    V = DFN_function_space(mesh)

    physical_setup = PhysicalSetup(V)

    dA_R = create_particle_summation_measure(mesh)

    num_particles = dfx.fem.assemble_scalar(
        dfx.fem.form(
            dfx.fem.Constant(mesh, 1.0) * dA_R
        )
    )

    A, a_ratios = physical_setup.total_surface_and_weights()
    L, Ls = physical_setup.mean_and_particle_affinities()

    assert np.isclose(A, 4 * np.pi * num_particles)

    assert np.isclose(dfx.fem.assemble_scalar(dfx.fem.form(a_ratios * dA_R)),
                      1.0)

    assert np.isclose(
        dfx.fem.assemble_scalar(dfx.fem.form(Ls * a_ratios * dA_R)),
        L)


def test_mesh():

    n_particles = 128

    mesh = create_1p1_DFN_mesh(comm, n_part=n_particles)
    dA = create_particle_summation_measure(mesh)

    form = dfx.fem.form(dfx.fem.Constant(mesh, 1.) * ufl.dx)
    value = dfx.fem.assemble_scalar(form)

    print(value)
    assert np.isclose(value, n_particles)

    form = dfx.fem.form(dfx.fem.Constant(mesh, 1.) * dA)
    value = dfx.fem.assemble_scalar(form)

    print(value)
    assert np.isclose(value, n_particles)

    r, _ = ufl.SpatialCoordinate(mesh)

    form = dfx.fem.form(r * dA)
    value = dfx.fem.assemble_scalar(form)

    print(value)
    assert np.isclose(value, n_particles)

    form = dfx.fem.form((1 - r) * dA)
    value = dfx.fem.assemble_scalar(form)

    print(value)
    assert np.isclose(value, 0.)


@pytest.mark.parametrize("I_global_value", [0., 1e-3, 0.1, 1.0, 10., 100.])
def test_Voltage_constant_mu(I_global_value: float):

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)
    W = scifem.create_real_functionspace(mesh)

    physical_setup = PhysicalSetup(V)

    # Test for mu = 0
    u = dfx.fem.Function(V)  # mu = 0
    voltage = dfx.fem.Function(W)
    v_voltage = ufl.TestFunction(W)
    dvoltage = ufl.TrialFunction(W)

    I_global = dfx.fem.Constant(mesh, I_global_value)

    voltage_ufl = voltage_form(u, voltage, v_voltage, I_global, physical_setup)

    dvoltage_ufl = ufl.derivative(voltage_ufl, voltage, dvoltage)

    solver = scifem.NewtonSolver([voltage_ufl], [[dvoltage_ufl]], [voltage])

    solver.solve()

    voltage_value = voltage.x.array[0]

    print(f"L * V(I={I_global.value}, mu=0) = ",
          physical_setup.mean_affinity * voltage_value)

    assert np.isclose(
        physical_setup.mean_affinity * voltage_value,
        -I_global.value)

    # Test for mu = 1
    _, mu = u.split()
    mu.x.array[:] = 1.

    solver.solve()
    voltage_value = voltage.x.array[0]

    assert np.isclose(
        physical_setup.mean_affinity * voltage_value,
        -I_global.value - physical_setup.mean_affinity)


def test_Voltage_constant_I_global():

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)
    W = scifem.create_real_functionspace(mesh)

    physical_setup = PhysicalSetup(V)

    u = dfx.fem.Function(V)
    voltage = dfx.fem.Function(W)
    v_voltage = ufl.TestFunction(W)
    dvoltage = ufl.TrialFunction(W)

    I_global = dfx.fem.Constant(mesh, 1.)

    voltage_ufl = voltage_form(u, voltage, v_voltage, I_global, physical_setup)

    dvoltage_ufl = ufl.derivative(voltage_ufl, voltage, dvoltage)

    solver = scifem.NewtonSolver([voltage_ufl], [[dvoltage_ufl]], [voltage])

    _, mu = u.split()

    for mu_value in [-100., -10., -1., 0., 1e-3, 0.1, 1.0, 10., 100.]:

        mu.x.array[:] = mu_value

        solver.solve()
        voltage_value = voltage.x.array[0]

        assert np.isclose(
            physical_setup.mean_affinity * voltage_value,
            -I_global.value - mu_value * physical_setup.mean_affinity)


def test_particle_current():

    mesh = create_1p1_DFN_mesh(comm)

    V = DFN_function_space(mesh)
    W = scifem.create_real_functionspace(mesh)

    u = dfx.fem.Function(V)

    voltage = dfx.fem.Function(W)
    v_voltage = ufl.TestFunction(W)
    dvoltage = ufl.TrialFunction(W)

    physical_setup = PhysicalSetup(V)

    # Randomize the particle chemical potentials to have
    # non-trivial particle current.
    u.x.array[:] = np.random.random(u.x.array.shape)

    I_global = dfx.fem.Constant(mesh, 0.)

    voltage_ufl = voltage_form(u, voltage, v_voltage, I_global, physical_setup)

    dvoltage_ufl = ufl.derivative(voltage_ufl, voltage, dvoltage)

    solver = scifem.NewtonSolver([voltage_ufl], [[dvoltage_ufl]], [voltage])

    test_current = TestCurrent(u, voltage, I_global, physical_setup)

    for I_global_value in [0., 1e-3, 0.1, 1.0, 10., 100.]:
        I_global.value = I_global_value

        solver.solve()

        I_global_ref = test_current.compute_current()

        print(I_global_value, I_global_ref)

        assert np.isclose(I_global_value, I_global_ref)
