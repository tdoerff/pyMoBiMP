"""
CH_4_min_blocked.py

Playground to develop a multi-particle integrator suitable for integrating
up to thousands of particles. The essential idea is that each particle problem
decouples from the other ones except for the boundary condition.

Open issues:
[ ] parallelization
[ ] runtime analysis
[ ] file output
[ ] implicit boundary condition?
"""

import dolfinx as dfx

from matplotlib import pyplot as plt

from mpi4py import MPI

import numpy as np

import os

import ufl

from pyMoBiMP.fenicsx_utils import (
    BlockNewtonSolver as BlockNewtonSolver,
    BlockNonlinearProblem,
    get_mesh_spacing,
    RuntimeAnalysisBase)
from pyMoBiMP.cahn_hilliard_utils import (
    cahn_hilliard_form,
    c_of_y,
    _free_energy as free_energy,
)


class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(self, u_states, filename):

        self.u_states = u_states
        self.filename = filename

        mesh = u_states[0].function_space.mesh
        coords = ufl.SpatialCoordinate(mesh)
        r2 = ufl.dot(coords, coords)

        self.mu_bc_forms = [dfx.fem.form(u.sub(1) * r2 * ufl.ds) for u in u_states]
        self.q_forms = [dfx.fem.form(3 * c_of_y(u.sub(0)) * r2 * ufl.dx)
                        for u in u_states]

    def analyze(self, t):

        mu_bcs = [dfx.fem.assemble_scalar(mu_bc_form)
                  for mu_bc_form in self.mu_bc_forms]
        qs = [dfx.fem.assemble_scalar(q_form) for q_form in self.q_forms]

        self.soc = sum(qs) / num_particles

        self.cell_voltage = compute_cell_voltage(I_total, L, mu_bcs, Ls, a_ratios)

        self.data.append([self.soc, self.cell_voltage])

        super().analyze(t)


def compute_cell_voltage(
    I_total: float,
    L: float,
    mu_bcs: list[float],
    Ls: list[float],
    a_ratios: list[float],
):

    weighted_particle_potentials = [
        L_k / L * a_k * mu_bc for L_k, a_k, mu_bc in zip(Ls, a_ratios, mu_bcs)
    ]

    active_phase_potential = sum(weighted_particle_potentials)

    cell_voltage = -(I_total / L + active_phase_potential)

    return cell_voltage


def log(*msg, cond=True):

    if cond:
        print("LOG: ", *msg, flush=True)


def warn(*msg):

    print("WARNING: ", *msg, flush=True)


def plot_solution(num_particles, T_final, us, figs_axs, t):
    for i_particle in range(num_particles):
        u = us[i_particle]
        V0, _ = u.function_space.sub(0).collapse()
        c = dfx.fem.Function(V0)

        fig, ax = figs_axs[i_particle]

        color = (min(t, T_final) / T_final, 0, 0)

        c_expr = dfx.fem.Expression(
                    c_of_y(u.sub(0).collapse()),
                    V0.element.interpolation_points())
        c.interpolate(c_expr)

        ax.plot(c.x.array[:], color=color)


if __name__ == "__main__":

    # Simulation setup
    # ----------------

    num_particles = 2
    T_final = 1.0
    I_total = 0.

    Ls = 1.e1 * (1. + 0.1 * (2. * np.random.random(num_particles) - 1.))
    a_ratios = np.ones(num_particles) / num_particles

    L = sum(_a * _L for _a, _L in zip(a_ratios, Ls))

    # Set up the mesh
    # ---------------
    comm = MPI.COMM_SELF

    mesh_filename = "Meshes/line_mesh.xdmf"
    with dfx.io.XDMFFile(comm, mesh_filename, 'r') as file:
        mesh = file.read_mesh(name="Grid")

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    # Set up function space
    # ---------------------

    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    elem_c, elem_mu = elem1, elem1

    # TODO: Is just one space enough?
    V = dfx.fem.FunctionSpace(mesh, elem_c * elem_mu)

    # Set up individual particle forms
    # --------------------------------

    dt = dfx.fem.Constant(mesh, 1e-9)

    i_ks = [dfx.fem.Constant(mesh, 0.0) for _ in range(num_particles)]

    Fs = []
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

    # Set up initial data (crudely simplified)
    # ----------------------------------------
    for u in u0s:
        u.sub(0).x.array[:] = -6.

    us[0].sub(0).x.array[:] = -5.5

    # Problem and solver setup
    # ------------------------
    problem = BlockNonlinearProblem(Fs, us)

    solver = BlockNewtonSolver(comm, problem, max_iterations=25, atol=1e-9)

    # # Do a single step to solve for mu
    its, success = solver.solve(us)

    assert success

    coords = ufl.SpatialCoordinate(mesh)
    r2 = ufl.dot(coords, coords)

    mu_bc_forms = [dfx.fem.form(u.sub(1) * r2 * ufl.ds) for u in us]

    q_forms = [dfx.fem.form(3 * c_of_y(u.sub(0)) * r2 * ufl.dx) for u in us]

    def callback(solver, us):

        mu_bcs = [dfx.fem.assemble_scalar(mu_bc_form)
                  for mu_bc_form in mu_bc_forms]

        cell_voltage = compute_cell_voltage(I_total, L, mu_bcs, Ls, a_ratios)

        # update particle currents
        for i_particle, _ in enumerate(us):
            i_ks[i_particle].value = \
                -Ls[i_particle] * (mu_bcs[i_particle] + cell_voltage)

        assert np.isclose(sum([i.value for i in i_ks]),  I_total)

    # attach to solver
    solver.callback = callback

    filename = os.path.dirname(os.path.abspath(__file__)) + "/rt.txt"
    rt_analysis = AnalyzeCellPotential(us, filename=filename)

    figs_axs = [plt.subplots() for _ in range(num_particles)]

    plot_solution(num_particles, T_final, us, figs_axs, 0.)

    t = 0.
    it = 0

    dt_min = 1e-9
    dt_max = 1e-3
    tol = 1e-5

    while t < T_final:

        if dt.value < dt_min:
            warn(f"Timestep too small! (dt={dt.value:1.3e})")

            break

        [u0.interpolate(u) for u0, u in zip(u0s, us)]

        mu_bcs = [dfx.fem.assemble_scalar(mu_bc_form) for mu_bc_form in mu_bc_forms]
        qs = [dfx.fem.assemble_scalar(q_form) for q_form in q_forms]

        soc = sum(qs) / num_particles

        cell_voltage = compute_cell_voltage(I_total, L, mu_bcs, Ls, a_ratios)

        try:
            iterations, success = solver.solve(us)
            assert success

        except (AssertionError, RuntimeError) as e:
            warn(e)

            # Reset and continue with a smaller time step
            [u.interpolate(u0) for u, u0 in zip(us, u0s)]
            dt.value *= 0.5

            warn(f"Reduce timestep size to dt={dt.value:1.3e}")

            continue

        except Exception as e:
            warn("!!! Uncought exception!")
            warn(e)

            break

        rt_analysis.analyze(t)

        # increase only after timestep was successfull.
        t += dt.value
        it += 1

        u_err_maxs = [np.abs(u.x.array - u0.x.array).max()
                      for u, u0 in zip(us, u0s)]

        u_err_max = max(u_err_maxs)

        # The new timestep size for the next timestep.
        dt.value = min(max(tol / u_err_max, dt_min), dt_max, 1.01 * dt.value)

        log(
            f"[{t/T_final * 100:>3.0f}%] " +
            f"t[{it:06}] = {t:2.4e} : " +
            f"dt = {dt.value:1.3e} ; " +
            f"iterations: {iterations} ; " +
            f"soc = {rt_analysis.soc:1.3e}",
            f"cell_voltage: {rt_analysis.cell_voltage}")

        # Output
        # ------
        if it % 100 == 0:

            plot_solution(num_particles, T_final, us, figs_axs, t)

    plt.show()
