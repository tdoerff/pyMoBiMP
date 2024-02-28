import dolfinx as dfx

import numpy as np

import ufl


def cahn_hilliard_form(psi, psi0, dt,
                       metric_factor=0,
                       M=lambda c: 1,
                       c_of_y=lambda y: y,
                       free_energy=lambda c: 0.25 * (c**2 - 1)**2,
                       lam=0.01,
                       I_charge=0.1,
                       theta=0.5):

    # TODO:
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
