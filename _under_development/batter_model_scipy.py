"""
battery_model_scipy.py

Dumpyard for old code that was used for a scipy-based time integration
implementation.
"""

def compute_dcdy(y, c_of_y):

    y = ufl.variable(y)
    c = c_of_y(y)

    dcdy = ufl.diff(c, y)

    return c, dcdy

def cahn_hilliard_mu_form(
        y,
        c_of_y=c_of_y,
        free_energy=_free_energy,
        gamma=0.1,
        form_weights=None):

    V = y.function_space
    mesh = V.mesh

    r = ufl.SpatialCoordinate(mesh)

    # adaptation of the volume element due to geometry
    if form_weights is not None:
        s_V = form_weights["volume"]
    else:
        s_V = 4 * np.pi * r**2

    c = c_of_y(y)
    mu_chem = compute_chemical_potential(free_energy, c)

    # adaptation of the volume element due to geometry
    if form_weights is not None:
        s_V = form_weights["volume"]
    else:
        s_V = 4 * np.pi * r**2

    mu_ = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    residual_mu = s_V * ufl.inner(mu_, v) * ufl.dx
    residual_mu -= s_V * ufl.inner(v, mu_chem) * ufl.dx + \
        gamma * s_V * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx

    return residual_mu


def cahn_hilliard_dydt_form(
        y,
        mu,
        I_charge,
        M=lambda c: c * (1 - c),
        c_of_y=c_of_y,
        form_weights=None):

    V = mu.function_space
    mesh = V.mesh

    r = ufl.SpatialCoordinate(mesh)

    # adaptation of the volume element due to geometry
    if form_weights is not None:
        s_V = form_weights["volume"]
        s_A = form_weights["surface"]
    else:
        s_V = 4 * np.pi * r**2
        s_A = 2 * np.pi * r**2

    c, dcdy = compute_dcdy(y, c_of_y)

    flux = M(c) * ufl.grad(mu)

    dydt = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    residual_dydt = s_V * dcdy * dydt * v * ufl.dx
    residual_dydt -= s_A * I_charge * v * ufl.ds
    residual_dydt -= -s_V * ufl.dot(ufl.grad(v), flux) * ufl.dx

    return residual_dydt