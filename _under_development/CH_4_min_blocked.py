import basix

import dolfinx
import dolfinx as dfx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver as NewtonSolverBase

from matplotlib import pyplot as plt

from mpi4py import MPI

import numpy as np

import petsc4py

import random

import petsc4py.PETSc
import ufl

from pyMoBiMP.fenicsx_utils import get_mesh_spacing
from pyMoBiMP.cahn_hilliard_utils import (
    cahn_hilliard_form,
    c_of_y,
    _free_energy as free_energy,
)


class NewtonSolver(NewtonSolverBase):
    def __init__(self, comm: MPI.Intracomm, problem: NonlinearProblem):
        super().__init__(comm, problem)

        # solver.krylov_solver.setType(PETSc.KSP.Type.PREONLY)
        # solver.krylov_solver.getPC().setType(PETSc.PC.Type.LU)
        ksp = self.krylov_solver
        opts = petsc4py.PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"  # TODO: direct solver! UMFPACK
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()

        self.max_it = 10
        self.rtol = 1e-3
        self.convergence_criterion = "incremental"


class SingleParticleProblem:

    def __init__(self, F, u, bcs=None, J=None, P=None):

        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        self._obj_vec = dolfinx.fem.petsc.create_vector(self._F)
        self._solution = u
        self._bcs = bcs
        self._P = P

        self.F_vec = dolfinx.fem.petsc.create_vector(self._F)
        self.J_mat = dolfinx.fem.petsc.create_matrix(self._J)

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guess
        provided in `self._solution`.
        """
        x = petsc4py.PETSc.Vec(self._solution.x.array.copy())
        with x.localForm() as _x, \
                petsc4py.PETSc.Vec(self._solution.x).localForm() as _solution:
            _x[:] = _solution
        return x

    def update_solution(self, x: petsc4py.PETSc.Vec) -> None:
        """Update `self._solution` with data in `x`."""
        x.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT,
            mode=petsc4py.PETSc.ScatterMode.FORWARD,
        )
        with x.localForm() as _x, self._solution.x.petsc_vec.localForm() as _solution:
            _solution[:] = _x

    def obj(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()

    def F(
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        F_vec: petsc4py.PETSc.Vec,
    ) -> None:
        """Assemble the residual."""
        self.update_solution(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F_vec, self._F)
        dolfinx.fem.petsc.apply_lifting(
            F_vec, [self._J], [self._bcs], x0=[x], scale=-1.0
        )
        F_vec.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE
        )
        dolfinx.fem.petsc.set_bc(F_vec, self._bcs, x, -1.0)

    def J(
        self,
        snes: petsc4py.PETSc.SNES,
        x: petsc4py.PETSc.Vec,
        J_mat: petsc4py.PETSc.Mat,
        P_mat: petsc4py.PETSc.Mat,
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J_mat, self._J, self._bcs, diagonal=1.0)
        J_mat.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(
                P_mat, self._P, self._bcs, diagonal=1.0
            )
            P_mat.assemble()


def compute_cell_voltage(
    I_total: float, L: float, mu_bcs: float, Ls: list[float], a_ratios: list[float]
):

    weighted_particle_potentials = [
        L_k / L * a_k * mu_bc for L_k, a_k, mu_bc in zip(Ls, a_ratios, mu_bcs)
    ]

    active_phase_potential = sum(weighted_particle_potentials)

    cell_voltage = -(I_total / L + active_phase_potential)

    return cell_voltage


if __name__ == "__main__":

    # Simulation setup
    # ----------------

    num_particles = 6
    T_final = .1
    I_total = 0.

    Ls = np.linspace(-1, 1, num_particles) + 1e1
    a_ratios = np.ones(num_particles) / num_particles

    L = sum(_a * _L for _a, _L in zip(a_ratios, Ls))

    # Set up the mesh
    # ---------------
    comm = MPI.COMM_SELF

    n_elem = 32
    mesh = dfx.mesh.create_unit_interval(comm, n_elem)
    print("create mesh.")

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    # Set up function space
    # ---------------------

    elem1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    elem_c, elem_mu = elem1, elem1

    # TODO: Is just one space enough?
    V = dfx.fem.FunctionSpace(mesh, elem_c * elem_mu)

    # Set up individual particle forms
    # --------------------------------

    dt = dfx.fem.Constant(mesh, 1e-4)

    i_ks = [dfx.fem.Constant(mesh, 0.0) for _ in range(num_particles)]

    Fs = []
    Js = [[None for _ in range(num_particles)] for _ in range(num_particles)]
    us = []
    u0s = []

    for i_particle in range(num_particles):

        u = dfx.fem.Function(V)
        u0 = dfx.fem.Function(V)
        du = ufl.TrialFunction(V)

        F = cahn_hilliard_form(
            u,
            u0,
            dt,
            M=lambda c: c * (1 - c),
            c_of_y=c_of_y,
            free_energy=free_energy,
            lam=0.1,
            I_charge=i_ks[i_particle],
        )

        J = ufl.derivative(F, u, du)

        us.append(u)
        u0s.append(u0)
        Fs.append(F)
        Js[i_particle][i_particle] = J

    # Set up initial data (crudely simplified)
    # ----------------------------------------
    for u in u0s:
        u.sub(0).x.array[:] = 2 * random.random() - 1

    # Problem and solver setup
    # ------------------------
    problems = [NonlinearProblem(F, u) for F, u, J in zip(Fs, us, Js)]

    solvers = [NewtonSolver(comm, problem) for problem in problems]

    # Do a single step to solve for mu
    [solver.solve(u) for solver, u in zip(solvers, us)]

    coords = ufl.SpatialCoordinate(mesh)
    r2 = ufl.dot(coords, coords)

    mu_bc_forms = [dfx.fem.form(u.sub(1) * r2 * ufl.ds) for u in us]

    figs_axs = [plt.subplots() for _ in range(num_particles)]

    t = 0.
    it = 0

    dt_min = 1e-9
    dt_max = 1e-3
    tol = 1e-7

    while t < T_final:

        it += 1

        mu_bcs = [dfx.fem.assemble_scalar(mu_bc_form) for mu_bc_form in mu_bc_forms]

        cell_voltage = compute_cell_voltage(I_total, L, mu_bcs, Ls, a_ratios)

        u_err_max = 0.
        iterations = 0

        for i_particle in range(num_particles):

            solver = solvers[i_particle]
            u = us[i_particle]
            u0 = u0s[i_particle]

            i_ks[i_particle].value = \
                - Ls[i_particle] * (mu_bcs[i_particle] + cell_voltage)

            u0.interpolate(u)

            it_part, _ = solver.solve(u)

            iterations = max(it_part, iterations)

            # Adaptive timestepping a la Yibao Li et al. (2017)
            u_max_loc = np.abs(u.x.array - u0.x.array).max()

            u_err_max += u_max_loc

        dt.value = min(max(tol / u_err_max, dt_min), dt_max, 1.01 * dt.value)

        t += dt.value

        print(
            f"[{t/T_final * 100:>3.0f}%] " +
            f"t[{it:06}] = {t:2.4e} : " +
            f"dt = {dt.value:1.3e} ; " +
            f"iterations: {iterations} ; ",
            f"cell_voltage: {cell_voltage}", flush=True)

        # Output
        # ------
        if it % 100 == 0:

            for i_particle in range(num_particles):

                u = us[i_particle]
                V0, _ = u.function_space.sub(0).collapse()
                c = dfx.fem.Function(V0)

                fig, ax = figs_axs[i_particle]

                color = (t / T_final, 0, 0)

                c_expr = dfx.fem.Expression(
                    c_of_y(u.sub(0).collapse()),
                    V0.element.interpolation_points())
                c.interpolate(c_expr)

                ax.plot(c.x.array[:], color=color)

    plt.show()
