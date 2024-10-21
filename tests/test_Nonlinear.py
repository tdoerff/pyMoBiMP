import dolfinx as dfx
from dolfinx.nls.petsc import NewtonSolver as NewtonOEM

from mpi4py.MPI import COMM_WORLD as comm, MIN, SUM

import numpy as np

from petsc4py import PETSc

import pytest

import ufl

from pyMoBiMP.fenicsx_utils import (
    NewtonSolver,
    NonlinearProblem,
)


def test_NonlinearProblem():
    """
    Test constom problem class against build-in solver
    to make sure we do not break the interface with the custom problem.
    """

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = (
        uh**2 * v * ufl.dx
        - 2 * uh * v * ufl.dx
        - (x[0] ** 2 + 4 * x[0] + 3) * v * ufl.dx
    )

    problem = NonlinearProblem(F, uh)

    solver = NewtonOEM(comm, problem)

    solver.solve(uh)

    def root_0(x):
        return 3 + x[0]

    def root_1(x):
        return -1 - x[0]

    u_ex0 = dfx.fem.Function(V)
    u_ex0.interpolate(lambda x: root_0(x))

    u_ex1 = dfx.fem.Function(V)
    u_ex1.interpolate(lambda x: root_1(x))

    L2_err0_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex0 - uh, u_ex0 - uh) * ufl.dx)
    )
    L2_err1_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex1 - uh, u_ex1 - uh) * ufl.dx)
    )

    L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)
    L2_err1 = mesh.comm.allreduce(L2_err1_loc, op=SUM)

    assert np.isclose(L2_err0, 0.0) or np.isclose(L2_err1, 0.0)


@pytest.mark.parametrize("order", [1, 2, 5, 9])
def test_nonlinear_algebraic(order):

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.functionspace(mesh, ("Lagrange", order))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = (x[0] - uh) ** 2 * v * ufl.dx

    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(lambda x: x[0])

    problem = NonlinearProblem(F, uh)

    solver = NewtonSolver(problem, max_iterations=50)

    solver.solve(uh)

    L2_err_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * ufl.dx)
    )

    L2_err0 = mesh.comm.allreduce(L2_err_loc, op=SUM)

    assert np.isclose(L2_err0, 0.0)


@pytest.mark.parametrize("order", [1,])
def test_differential(order):

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.functionspace(mesh, ("Lagrange", order))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = (
        ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        - 2 * x[0] * v * ufl.ds
        + 2 * v * ufl.dx
    )

    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(lambda x: x[0] ** 2)

    problem = NonlinearProblem(F, uh)

    solver = NewtonSolver(problem, max_iterations=100)

    const = dfx.fem.Function(V)
    const.interpolate(lambda x: np.ones_like(x[0]))

    C = dfx.fem.petsc.assemble_vector(dfx.fem.form(const * v * ufl.dx))
    C.scale(1.0 / C.norm())

    assert np.isclose(C.norm(), 1.0)

    # Create the PETSc nullspace vector and check
    # that it is a valid nullspace of A
    nullspace = PETSc.NullSpace().create(vectors=[C], comm=mesh.comm)
    solver.A.setNullSpace(nullspace)

    # assert nullspace.test(solver.A)

    ksp = solver.ksp

    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    # do not compute null space again
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    ksp.getPC().setFactorSetUpSolverType()

    # solver.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    # solver.A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

    _, success = solver.solve(uh)

    assert success

    min_uh_loc = uh.x.array.min()
    min_uh = mesh.comm.allreduce(min_uh_loc, op=MIN)

    uh.x.array[:] -= min_uh

    L2_err0_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * ufl.dx)
    )
    L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)

    assert np.isclose(L2_err0, 0.0)
