import basix

import dolfinx as dfx
import dolfinx.fem.petsc

from mpi4py import MPI

import numpy as np

import ufl


def _free_energy(
    u: dfx.fem.Function, a: float = 6.0 / 4.0, b: float = 0.2, c: float = 5.0
):

    fe = (
        u * ufl.ln(u)
        + (1 - u) * ufl.ln(1 - u)
        + a * u * (1 - u)
        + b * ufl.sin(c * np.pi * u)
    )

    return fe


# Forward and backward variable transformation.
def c_of_y(y):
    return ufl.exp(y) / (1 + ufl.exp(y))


def y_of_c(c):
    return ufl.ln(c / (1 - c))


def compute_chemical_potential(free_energy, c):

    # Differentiate the free energy function to
    # obtain the chemical potential
    c = ufl.variable(c)
    dfdc = ufl.diff(free_energy(c), c)
    mu_chem = dfdc
    return mu_chem


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


class ParticleCurrentDensity(dfx.fem.Constant):
    def __init__(self, comm, mesh, A_k, L_k, I_charge):

        super().__init__(mesh, 0.0)

        self.comm = comm
        self.mesh = mesh

        coords = ufl.SpatialCoordinate(mesh)
        self.r = ufl.sqrt(sum([co**2 for co in coords]))

        self.A_k = A_k
        self.L_k = L_k

        self.A = A = comm.allreduce(A_k, op=MPI.SUM)

        # fraction of the total surface
        self.a_k = a_k = A_k / A

        # Coupling parameters between particle surface potential.
        self.L = comm.allreduce(a_k * L_k, op=MPI.SUM)

        self.I_charge = I_charge

    def attach_mu(self, mu_k):

        self.mu_k = mu_k

    @property
    def value(self):

        # CAUTION: This trick only works if R=1!
        # TODO: Replace by more robust expression.
        mu_bc = dfx.fem.assemble_scalar(dfx.fem.form(self.mu_k * self.r**2 * ufl.ds))

        # I * (A_1 + A_2) = I_1 * A_1 + I_2 * A_2
        term = self.comm.allreduce(self.L_k * self.a_k * mu_bc, op=MPI.SUM)

        # Here must be a negative sign since with the I_charges, we measure
        # what flows out of the particle.
        Voltage = - (self.I_charge.value + term) / self.L

        # TODO: Check the sign! Somehow, there must be a minus for the
        # code to work. I think, I_charge as constructed here is the current
        # OUT OF the particle.
        I_charge_k = - self.L_k * (mu_bc + Voltage)

        # I think that's the magic: Inplace-copying the current to
        # _cpp_object.value updates the Constant object everytime the
        # property method value() is called.
        np.copyto(self._cpp_object.value, np.asarray(I_charge_k))

        return I_charge_k


def create_1p1_DFN_mesh(comm, n_rad=16, n_part=192):

    if comm.rank == 0:
        radial_grid = np.linspace(0, 1, n_rad)
        particle_grid = np.linspace(0, 1, n_part)

        rr, pp = np.meshgrid(radial_grid, particle_grid)

        coords_grid = np.stack((rr, pp)).transpose((-1, 1, 0)).copy()

        coords_grid.shape

        coords_grid_flat = coords_grid.reshape(-1, 2).copy()

        # All the radial connections
        elements_radial = [
            [[n_part * i + k, n_part * (i + 1) + k] for i in range(n_rad - 1)]
            for k in range(n_part)
        ]

        elements_radial = np.array(elements_radial).reshape(-1, 2)

        # Connections between particles
        elements_bc = (n_rad - 1) * n_part + np.array(
            [[k, k + 1] for k in range(n_part - 1)]
        )
        elements_bc = []  # With elements at the outer edge the integration fails.

        elements = np.array(list(elements_bc) + list(elements_radial))

    else:
        coords_grid_flat = np.empty((0, 2), dtype=np.float64)
        elements = np.empty((0, 2), dtype=np.int64)

    gdim = 2
    shape = "interval"
    degree = 1

    domain = ufl.Mesh(basix.ufl.element("Lagrange", shape, degree, shape=(gdim,)))

    mesh = dfx.mesh.create_mesh(comm, elements[:, :gdim], coords_grid_flat, domain)
    return mesh


def create_particle_summation_measure(mesh):
    # TODO: rename to particle_summation_measure
    # %% Create integral measure on the particle surface
    # --------------------------------------------------
    fdim = mesh.topology.dim - 1

    facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 1.))

    facet_markers = np.full_like(facets, 1)

    facet_tag = dfx.mesh.meshtags(mesh, fdim, facets, facet_markers)

    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
    dA_R = dA(1)

    return dA_R
