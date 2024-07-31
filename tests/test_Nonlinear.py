import dolfinx as dfx
from dolfinx.nls.petsc import NewtonSolver as NewtonOEM

from mpi4py.MPI import COMM_WORLD as comm, MIN, SUM

import numpy as np

from petsc4py import PETSc

import pytest

import ufl

from pyMoBiMP.fenicsx_utils import (
    evaluation_points_and_cells,
    NewtonSolver,
    NonlinearProblem,
    BlockNewtonSolver,
    BlockNonlinearProblem,
)


def test_NonlinearProblem():
    """
    Test constom problem class against build-in solver
    to make sure we do not break the interface with the custom problem.
    """

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.FunctionSpace(mesh, ("Lagrange", 1))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = uh**2 * v * ufl.dx - 2 * uh * v * ufl.dx - \
        (x[0]**2 + 4 * x[0] + 3) * v * ufl.dx

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
        dfx.fem.form(ufl.inner(u_ex0 - uh, u_ex0 - uh) * ufl.dx))
    L2_err1_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex1 - uh, u_ex1 - uh) * ufl.dx))

    L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)
    L2_err1 = mesh.comm.allreduce(L2_err1_loc, op=SUM)

    assert np.isclose(L2_err0, 0.) or np.isclose(L2_err1, 0.)


@pytest.mark.parametrize("order", [1, 2, 5, 9])
def test_nonlinear_algebraic(order):

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.FunctionSpace(mesh, ("Lagrange", order))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = (x[0] - uh)**2 * v * ufl.dx

    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(lambda x: x[0])

    problem = NonlinearProblem(F, uh)

    solver = NewtonSolver(comm, problem, max_iterations=50)

    solver.solve(uh)

    L2_err_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * ufl.dx))

    L2_err0 = mesh.comm.allreduce(L2_err_loc, op=SUM)

    assert np.isclose(L2_err0, 0.)


@pytest.mark.parametrize("order", [1, 5])
def test_differential(order):

    mesh = dfx.mesh.create_unit_interval(comm, 128)

    V = dfx.fem.FunctionSpace(mesh, ("Lagrange", order))

    uh = dfx.fem.Function(V)

    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - \
        2 * x[0] * v * ufl.ds + \
        2 * v * ufl.dx

    u_ex = dfx.fem.Function(V)
    u_ex.interpolate(lambda x: x[0]**2)

    problem = NonlinearProblem(F, uh)

    solver = NewtonSolver(comm, problem, max_iterations=100)

    const = dfx.fem.Function(V)
    const.interpolate(lambda x: np.ones_like(x[0]))

    C = dfx.fem.petsc.assemble_vector(dfx.fem.form(const * v * ufl.dx))
    C.scale(1. / C.norm())

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
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.getPC().setFactorSetUpSolverType()

    # solver.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    # solver.A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

    _, success = solver.solve(uh)

    assert success

    min_uh_loc = uh.x.array.min()
    min_uh = mesh.comm.allreduce(min_uh_loc, op=MIN)

    uh.x.array[:] -= min_uh

    L2_err0_loc = dfx.fem.assemble_scalar(
        dfx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * ufl.dx))
    L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)

    assert np.isclose(L2_err0, 0.)


def NonlinearBlockProblemCreation_algebraic():
    """
    Test constom problem class against build-in solver
    to make sure we do not break the interface with the custom problem.
    """

    meshes = [dfx.mesh.create_interval(comm, 128, [-1, 0.]),
              dfx.mesh.create_unit_interval(comm, 128)]

    def set_up_u(mesh):
        V = dfx.fem.FunctionSpace(mesh, ("Lagrange", 1))

        u = dfx.fem.Function(V)

        return u

    us = [set_up_u(mesh) for mesh in meshes]

    def set_up_form(u):

        V = u.function_space
        mesh = V.mesh

        x = ufl.SpatialCoordinate(mesh)

        v = ufl.TestFunction(V)
        F = (u - x[0])**2 * v * ufl.dx

        return F

    def u_exact(x):
        return x

    Fs = [set_up_form(u) for u in us]

    problem = BlockNonlinearProblem(Fs, us)

    return us, problem, u_exact


def test_nonlinear_problem_creation():
    _ = NonlinearBlockProblemCreation_algebraic()


def test_nonlinear_block_algebraic():

    us, block_problem, u_exact = NonlinearBlockProblemCreation_algebraic()

    solver = BlockNewtonSolver(comm, block_problem, convergence_criterion="residual")

    solver.solve(us)

    for u in us:

        V = u.function_space
        mesh = V.mesh

        u_ex = dfx.fem.Function(V)
        u_ex.interpolate(lambda x: u_exact(x[0]))

        L2_err0_loc = dfx.fem.assemble_scalar(
            dfx.fem.form(ufl.inner(u_ex - u, u_ex - u) * ufl.dx))
        L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)

        assert np.isclose(L2_err0, 0.)


def test_nonlinear_block_differential():

    # Adjacent meshes
    meshes = [dfx.mesh.create_interval(comm, 128, [-1, 0.]),
              dfx.mesh.create_unit_interval(comm, 128)]

    def set_up_u(mesh):
        V = dfx.fem.FunctionSpace(mesh, ("Lagrange", 4))

        u = dfx.fem.Function(V)

        return u

    us = [set_up_u(mesh) for mesh in meshes]

    def set_up_form(u):

        V = u.function_space
        mesh = V.mesh

        x_min = mesh.geometry.x[:, 0].min()
        x_max = mesh.geometry.x[:, 0].max()

        u_left = dfx.fem.Constant(mesh, 0.,)
        u_right = dfx.fem.Constant(mesh, 0.,)

        dofs_left = dfx.fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], x_min))

        dofs_right = dfx.fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], x_max))

        bcs = [dfx.fem.dirichletbc(u_left, dofs_left, V),
               dfx.fem.dirichletbc(u_right, dofs_right, V)]

        v = ufl.TestFunction(V)
        F = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - 2 * v * ufl.dx

        return F, u, bcs, u_left, u_right

    problem_defs = [set_up_form(u) for u in us]

    Fs = [F for F, *_ in problem_defs]
    us = [u for _, u, *_ in problem_defs]
    bcs = [bc for _, _, bc, *_ in problem_defs]
    u_lefts = [u_left for _, _, _, u_left, _ in problem_defs]
    u_rights = [u_right for _, _, _, _, u_right in problem_defs]

    # Manually set the boundary conditions for both grids.
    u_lefts[0].value = 2.
    u_rights[0].value = 1.
    u_lefts[1].value = 1.
    u_rights[1].value = 2.

    problem = BlockNonlinearProblem(Fs, us, bcs)

    coords = ufl.SpatialCoordinate(meshes[1])
    r2 = ufl.dot(coords, coords)

    # The term (1 - r2) selects the inner interface.
    u_1_l_form = dfx.fem.form(us[1] * (1 - r2) * ufl.ds)
    u_0_r_form = dfx.fem.form(us[0] * (1 - r2) * ufl.ds)

    def callback(solver, uhs):
        """
        Assign the boundary inner value of the left domain to right one
        and vice versa.
        """

        # First function
        # ---------------
        u_1_l = dfx.fem.assemble_scalar(u_1_l_form)

        u_rights[0].value = u_1_l

        # # Second function
        # # ---------------
        u_0_r = dfx.fem.assemble_scalar(u_0_r_form)

        u_lefts[1].value = u_0_r

    solver = BlockNewtonSolver(comm, problem, callback=callback)

    solver.solve(us)

    assert True
