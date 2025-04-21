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

    solver = NewtonSolver(comm, problem, max_iterations=50)

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

    solver = NewtonSolver(comm, problem, max_iterations=100)

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


def test_DirichletBCs():

    # Mesh and function space
    # =======================
    mesh = dfx.mesh.create_rectangle(
        comm,
        ((-1, -1), (1, 1)),
        (32, 32)
        )

    V = dfx.fem.functionspace(mesh, ("Lagrange", 4))

    # Solution and test function
    # ==========================
    uh = dfx.fem.Function(V)
    v = ufl.TestFunction(V)

    # Exact solution
    # ==============
    x, y = ufl.SpatialCoordinate(mesh)

    u_ex_expr = dfx.fem.Expression(
        100 + x**2 + y**2, V.element.interpolation_points()
    )

    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(u_ex_expr)  # type: ignore

    # Boundary conditions
    # ===================
    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_connectivity(fdim, tdim)

    boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dfx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    bcs = [dfx.fem.dirichletbc(u_ex, boundary_dofs)]  # type: ignore

    # Weak FEM form
    # =============
    F = ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx +  2 * tdim * v * ufl.dx  # type: ignore

    # Problem and solver
    # ==================

    problem = NonlinearProblem(F, uh, bcs=bcs)  # type: ignore
    solver = NewtonSolver(comm, problem, max_iterations=100)

    it, success = solver.solve(uh)  # type: ignore

    assert it < 3

    # L2 error
    # ========
    L2_err_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * ufl.dx)  # type: ignore
    )

    L2_err = mesh.comm.allreduce(L2_err_loc, op=SUM)

    print("L2 error: ", L2_err)

    assert np.isclose(L2_err, 0.0)
