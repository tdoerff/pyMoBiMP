import dolfinx as dfx

from mpi4py import MPI

import numpy as np

import ufl

from fenicsx_utils import (evaluation_points_and_cells,
                           RuntimeAnalysisBase,
                           StopEvent)


def cahn_hilliard_form(psi, psi0, dt,
                       M=lambda c: 1,
                       c_of_y=lambda y: y,
                       free_energy=lambda c: 0.25 * (c**2 - 1)**2,
                       lam=0.01,
                       I_charge=0.1,
                       theta=0.5):

    #   [ ] Assert whether psi, psi0, and v are on the same mesh/V
    V = psi.function_space
    mesh = V.mesh

    # Split the functions
    y, mu = ufl.split(psi)
    y0, mu0 = ufl.split(psi0)

    y = ufl.variable(y)
    c = c_of_y(y)

    dcdy = ufl.diff(c, y)

    v_c, v_mu = ufl.TestFunctions(V)

    # Differentiate the free energy function to
    # obtain the chemical potential
    c = ufl.variable(c)
    dfdc = ufl.diff(free_energy(c), c)
    mu_chem = dfdc

    # Theta scheme
    theta = dfx.fem.Constant(mesh, theta)

    mu_theta = theta * mu + (theta - 1.) * mu0

    r = ufl.SpatialCoordinate(mesh)

    # adaptation of the volume element due to geometry
    s_V = 4 * np.pi * r**2
    s_A = 2 * np.pi * r**2

    dx = ufl.dx  # The volume element
    ds = ufl.ds  # The surface element

    flux = M(c) * ufl.grad(mu_theta)

    F1 = s_V * dcdy * (y - y0) * v_c * dx
    F1 += s_V * ufl.dot(flux, ufl.grad(v_c)) * dt * dx
    F1 -= s_A * I_charge * v_c * dt * ds

    F2 = s_V * mu * v_mu * dx
    F2 -= s_V * mu_chem * v_mu * dx
    F2 -= s_V * ufl.inner(lam * ufl.grad(c), ufl.grad(v_mu)) * dx

    F = F1 + F2

    return F


def charge_discharge_stop(t, u, I_charge, c_bounds=[0.05, 0.99], c_of_y=lambda y: y):

    V = u.function_space

    mesh = V.mesh

    W, dof = V.sub(0).collapse()

    y, mu = u.split()

    c = dfx.fem.Function(W)

    c.interpolate(dfx.fem.Expression(c_of_y(y), W.element.interpolation_points()))

    max_c = mesh.comm.allreduce(max(c.x.array), op=MPI.MAX)
    min_c = mesh.comm.allreduce(min(c.x.array), op=MPI.MIN)

    print(f"t={t:1.5f} ; min_c = {min_c:1.3e} ; max_c = {max_c:1.3e}")

    if max_c > c_bounds[1] and I_charge.value > 0.0:
        print(
            f">>> total charge exceeds maximum (max(c) = {max_c:1.3f} > {c_bounds[0]:1.3f})."
        )
        print(">>> Start discharging.")
        I_charge.value *= -1.0

        return False

    if min_c < c_bounds[0] and I_charge.value < 0.0:

        print("Particle is emptied!")

        return True

    return False


class AnalyzeOCP(RuntimeAnalysisBase):

    def setup(
        self, *args, c_of_y=lambda y: y, free_energy=lambda u: 0.5 * u**2, **kwargs
    ):
        self.free_energy = free_energy
        self.c_of_y = c_of_y

        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):

        V = u_state.function_space
        mesh = V.mesh

        y, mu = u_state.split()

        c = self.c_of_y(y)

        r = ufl.SpatialCoordinate(mesh)

        c = ufl.variable(c)
        dFdc = ufl.diff(self.free_energy(c), c)

        charge = dfx.fem.form(3 * c * r**2 * ufl.dx)
        charge = dfx.fem.assemble_scalar(charge)

        chem_pot = dfx.fem.form(3 * dFdc * r**2 * ufl.dx)
        chem_pot = dfx.fem.assemble_scalar(chem_pot)

        x, cell = evaluation_points_and_cells(mesh, np.array([1.0]))

        mu_bc = float(mu.eval(x, cell))

        self.data.append([charge, chem_pot, mu_bc])

        return super().analyze(u_state, t)
