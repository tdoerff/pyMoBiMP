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
    MultiBlockNewtonSolver,
    MultiBlockNonlinearProblem,
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


def NonlinearBlockProblemCreation_algebraic():
    """
    Test constom problem class against build-in solver
    to make sure we do not break the interface with the custom problem.
    """

    meshes = [
        dfx.mesh.create_interval(comm, 128, [-1, 0.0]),
        dfx.mesh.create_unit_interval(comm, 128),
    ]

    def set_up_u(mesh):
        V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

        u = dfx.fem.Function(V)

        return u

    us = [set_up_u(mesh) for mesh in meshes]

    def set_up_form(u):

        V = u.function_space
        mesh = V.mesh

        x = ufl.SpatialCoordinate(mesh)

        v = ufl.TestFunction(V)
        F = (u - x[0]) ** 2 * v * ufl.dx

        return F

    def u_exact(x):
        return x

    Fs = [set_up_form(u) for u in us]

    problem = MultiBlockNonlinearProblem(Fs, us)

    return us, problem, u_exact


def test_nonlinear_problem_creation():
    _ = NonlinearBlockProblemCreation_algebraic()


def test_nonlinear_block_algebraic():

    us, block_problem, u_exact = NonlinearBlockProblemCreation_algebraic()

    solver = MultiBlockNewtonSolver(
        comm, block_problem, convergence_criterion="residual", atol=1e-10, rtol=0
    )

    solver.solve(us)

    for u in us:

        V = u.function_space
        mesh = V.mesh

        u_ex = dfx.fem.Function(V)
        u_ex.interpolate(lambda x: u_exact(x[0]))

        L2_err0_loc = dfx.fem.assemble_scalar(
            dfx.fem.form(ufl.inner(u_ex - u, u_ex - u) * ufl.dx)
        )
        L2_err0 = mesh.comm.allreduce(L2_err0_loc, op=SUM)

        assert np.isclose(L2_err0, 0.0)


def test_nonlinear_block_differential():
    r"""
    The purpose of this test is to reproduce a solution of the elliptic problem
    $$
        u'' = 2
    $$

    on $\Omega = [-1, 1]$ with $u(-1) = u(1) = 2$. The solution domain is split into
    $\Omega_L = [-1, 0]$ and $\Omega_R = [0, 1]$ and the matching conditions is
    $\lim_{\Omega_L \ni x\nearrow 0} u'(x) = lim_{\Omega_R \ni x\searrow 0} u'(x)$
    """

    # Adjacent meshes
    meshes = [
        dfx.mesh.create_interval(comm, 128, [-1, 0.0]),
        dfx.mesh.create_unit_interval(comm, 128),
    ]

    def set_up_u(mesh):
        V = dfx.fem.functionspace(mesh, ("Lagrange", 4))

        u = dfx.fem.Function(V)

        return u

    us = [set_up_u(mesh) for mesh in meshes]

    def set_up_form(u):

        V = u.function_space
        mesh = V.mesh

        # Constants to access the boundary conditions at runtime.
        u_inner = dfx.fem.Constant(
            mesh,
            0.0,
        )
        u_outer = dfx.fem.Constant(
            mesh,
            0.0,
        )

        # Set out Dirichlet BC.
        dofs_outer = dfx.fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0] ** 2, 1.0)
        )

        bcs = [dfx.fem.dirichletbc(u_outer, dofs_outer, V)]

        coords = ufl.SpatialCoordinate(mesh)
        r2 = ufl.dot(coords, coords)

        # The form including inner Neumann BCs.
        v = ufl.TestFunction(V)
        F = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + 2 * v * ufl.dx
        F -= u_inner * v * (1.0 - r2) * ufl.ds

        return F, u, bcs, u_inner, u_outer

    problem_defs = [set_up_form(u) for u in us]

    Fs = [F for F, *_ in problem_defs]
    us = [u for _, u, *_ in problem_defs]
    bcs = [bc for _, _, bc, *_ in problem_defs]
    u_inners = [u_inner for _, _, _, u_inner, _ in problem_defs]
    u_outers = [u_outer for _, _, _, _, u_outer in problem_defs]

    # Manually set the boundary conditions for both grids.
    u_outers[0].value = 2.0
    u_outers[1].value = 2.0

    # Manually perturb the inner Neumann BC.
    u_inners[0].value = 1.0
    u_inners[1].value = 1.0

    problem = MultiBlockNonlinearProblem(Fs, us, bcs)

    coords = ufl.SpatialCoordinate(meshes[1])
    r2 = ufl.dot(coords, coords)
    n = ufl.FacetNormal(meshes[1])

    # The term (1 - r2) selects the inner interface.
    du_1_l_form = dfx.fem.form(ufl.dot(ufl.grad(us[1]), n) * (1 - r2) * ufl.ds)

    coords = ufl.SpatialCoordinate(meshes[0])
    r2 = ufl.dot(coords, coords)
    n = ufl.FacetNormal(meshes[0])

    du_0_r_form = dfx.fem.form(ufl.dot(ufl.grad(us[0]), n) * (1 - r2) * ufl.ds)

    def callback(solver, uhs):
        """
        Assign the boundary inner value of the left domain to right one
        and vice versa.
        """

        # First function
        # ---------------
        du_1_l = dfx.fem.assemble_scalar(du_1_l_form)

        u_inners[0].value = du_1_l

        # # Second function
        # # ---------------
        du_0_r = dfx.fem.assemble_scalar(du_0_r_form)

        u_inners[1].value = du_0_r

    solver = MultiBlockNewtonSolver(comm, problem, callback=callback)

    solver.solve(us)

    def u_ex_fun(x):
        return x[0] ** 2 + 1.0

    u_exs = [set_up_u(mesh) for mesh in meshes]

    u_exs[0].interpolate(u_ex_fun)
    u_exs[1].interpolate(u_ex_fun)

    L2_err_locs = [
        dfx.fem.assemble_scalar(dfx.fem.form(ufl.inner(u_ex - u, u_ex - u) * ufl.dx))
        for u, u_ex in zip(us, u_exs)
    ]

    L2_err = sum(
        [
            mesh.comm.allreduce(L2_err_loc, op=SUM)
            for mesh, L2_err_loc in zip(meshes, L2_err_locs)
        ]
    )

    assert np.isclose(L2_err, 0.0)
