from collections.abc import Callable

import dolfinx as dfx

from dolfinx.fem.petsc import NonlinearProblem

from mpi4py import MPI

import numpy as np

import os

import pathlib

from typing import Optional, TypedDict, Unpack

import ufl

from fenicsx_utils import (evaluation_points_and_cells,
                           get_mesh_spacing,
                           NewtonSolver,
                           RuntimeAnalysisBase,
                           time_stepping,
                           Fenicx1DOutput)


# Forward and backward variable transformation.
def c_of_y(y):
    return ufl.exp(y) / (1 + ufl.exp(y))


def y_of_c(c):
    return ufl.ln(c / (1 - c))


def cahn_hilliard_form(
    psi,
    psi0,
    dt,
    M=lambda c: 1,
    c_of_y=c_of_y,
    free_energy=lambda c: 0.25 * (c**2 - 1) ** 2,
    lam=0.01,
    I_charge=0.1,
    theta=0.5,
    form_weights=None
):

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

    mu_theta = theta * mu + (theta - 1.0) * mu0

    r = ufl.SpatialCoordinate(mesh)

    # adaptation of the volume element due to geometry
    if form_weights is not None:
        s_V = form_weights["volume"]
        s_A = form_weights["surface"]
    else:
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


def populate_initial_data(u_ini, c_ini_fun, free_energy, y_of_c=y_of_c):

    # Store concentration-like quantity into state vector
    # ---------------------------------------------------

    V = u_ini.function_space

    W = V.sub(1).collapse()[0]
    c_ini = dfx.fem.Function(W)
    c_ini.interpolate(c_ini_fun)

    y_ini = dfx.fem.Expression(y_of_c(c_ini), W.element.interpolation_points())

    u_ini.sub(0).interpolate(y_ini)

    # Store chemical potential into state vector
    # ------------------------------------------

    W = u_ini.sub(1).function_space.element
    c_ini = ufl.variable(c_ini)
    dFdc = ufl.diff(free_energy(c_ini), c_ini)

    u_ini.sub(1).interpolate(dfx.fem.Expression(dFdc, W.interpolation_points()))

    u_ini.x.scatter_forward()


def charge_discharge_stop(
    t,
    u,
    I_charge,
    c_bounds=[0.05, 0.99],
    c_of_y=c_of_y,
    stop_at_empty=True,
    cycling=True,
    logging=False
):

    coords = ufl.SpatialCoordinate(u.function_space.mesh)
    r = ufl.sqrt(sum([c for c in coords]))

    y, _ = u.split()

    c = c_of_y(y)

    c_bc = dfx.fem.form(r**2 * c * ufl.ds)
    c_bc = dfx.fem.assemble_scalar(c_bc)

    if logging:
        print(f"t={t:1.5f} ; c_bc = {c_bc:1.3e}", c_bounds)

    if c_bc > c_bounds[1] and I_charge.value > 0.0:
        print(
            f">>> charge at boundary exceeds maximum (max(c) = {c_bc:1.3f} > {c_bounds[0]:1.3f})."
        )
        print(">>> Start discharging.")
        I_charge.value *= -1.0

        return False

    if c_bc < c_bounds[0] and I_charge.value < 0.0:

        if stop_at_empty:
            print("Particle is emptied!")

            return True

        else:
            if cycling:
                print(">>> Start charging.")
                I_charge.value *= -1.0

            else:
                print(">>> Stop charging.")
                I_charge.value = 0.0

            return False

    return False


class AnalyzeOCP(RuntimeAnalysisBase):

    def setup(
        self,
        *args,
        c_of_y=c_of_y,
        free_energy=lambda u: 0.5 * u**2,
        filename=None,
        **kwargs,
    ):
        self.free_energy = free_energy
        self.c_of_y = c_of_y

        self.filename = filename

        if self.filename is not None:
            with open(self.filename, "w") as file:
                pass

        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):

        V = u_state.function_space
        mesh = V.mesh

        y, mu = u_state.split()

        c = self.c_of_y(y)

        coords = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(sum([co for co in coords]))

        c = ufl.variable(c)
        dFdc = ufl.diff(self.free_energy(c), c)

        charge = dfx.fem.form(3 * c * r**2 * ufl.dx)
        charge = dfx.fem.assemble_scalar(charge)

        chem_pot = dfx.fem.form(3 * dFdc * r**2 * ufl.dx)
        chem_pot = dfx.fem.assemble_scalar(chem_pot)

        mu_bc = dfx.fem.form(mu * r**2 * ufl.ds)
        mu_bc = dfx.fem.assemble_scalar(mu_bc)

        self.data.append([charge, chem_pot, mu_bc])

        if self.filename is not None:
            with open(self.filename, "a") as file:
                np.savetxt(file, np.array([[t, charge, chem_pot, mu_bc]]))

        return super().analyze(u_state, t)


# Defaults for the Simulation class
_mesh = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 128)


def _free_energy(
    u: dfx.fem.Function, a: float = 6.0 / 4.0, b: float = 0.2, c: float = 5.0
):

    fe = (
        u * ufl.ln(u) \
        + (1 - u) * ufl.ln(1 - u) \
        + a * u * (1 - u) \
        + b * ufl.sin(c * np.pi * u) \
    )

    return fe

# [ ] make it possible to hand over pre-configured output class or object
# [ ] add runtime analysis

class Simulation:

    def __init__(
        self,
        mesh: dfx.mesh.Mesh = _mesh,
        element: Optional[ufl.FiniteElement | ufl.MixedElement] = None,
        free_energy: Callable[[dfx.fem.Function], dfx.fem.Expression] = _free_energy,
        T_final: float = 2.0,
        experiment: Callable[
            [float, dfx.fem.Function], dfx.fem.Expression
        ] = charge_discharge_stop,
        gamma: float = 0.1,
        M: Callable[
            [dfx.fem.Function | dfx.fem.Expression], dfx.fem.Expression
        ] = lambda c: 1.0 * c * (1 - c),
        I: float = 1.0,
        eps: float = 1e-3,
        c_ini=lambda x, eps: eps * np.ones_like(x[0]),
        output_file: Optional[str | os.PathLike] = None,
        n_out: int = 51,
        runtime_analysis: Optional[RuntimeAnalysisBase] = None,
        logging: bool=True
    ):

        # Define mixed element and function space
        # ---------------------------------------
        if element is None:
            element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        if element.num_sub_elements() == 2:
            mixed_element = element
        elif element.num_sub_elements() > 2:
            raise ValueError(
                f"element.num_sub_elements() = {element.num_sub_elements()} > 2"
            )
        else:
            mixed_element = element * element

        V = dfx.fem.FunctionSpace(mesh, mixed_element)

        # (initial) timestep
        # ------------------
        dx = get_mesh_spacing(mesh)
        self.dt = dfx.fem.Constant(mesh, 1e-2 * dx / I)

        self.T_final = T_final

        # The mixed-element functions
        # ---------------------------
        self.u = dfx.fem.Function(V)
        self.u0 = dfx.fem.Function(V)

        # Experimental setup
        # ------------------

        # Charging current goes into the form and is recomputed in experiment
        I_charge = dfx.fem.Constant(mesh, I)

        self.event_params = dict(I_charge=I_charge)

        self.experiment = experiment

        # The weak form of the problem
        # ----------------------------
        F = cahn_hilliard_form(
            self.u,
            self.u0,
            self.dt,
            free_energy=free_energy,
            theta=0.75,
            c_of_y = lambda y: c_of_y(y),
            M=M,
            lam=gamma,
            **self.event_params,
        )

        # Initial data
        # ------------

        populate_initial_data(self.u,
                              lambda r: c_ini(r, eps),
                              free_energy)

        # Problem and solver setup
        # ------------------------
        problem = NonlinearProblem(F, self.u)

        self.solver = NewtonSolver(mesh.comm, problem)

        # Setup output
        # ------------

        if output_file is not None:
            # make sure directory exists.
            real_file_path = os.path.realpath(output_file)

            dir_path = os.path.dirname(real_file_path)
            dir_path = pathlib.Path(dir_path)

            dir_path.mkdir(exist_ok=True, parents=True)

            # FIXME: This is jsut a dirty hack to get some output running.
            # Write a consistent file writer class and use here!!!
            self.output = Fenicx1DOutput(
                self.u,
                np.linspace(0, T_final, n_out),
                np.linspace(0, 1., 101))
        else:
            self.output = None

        self.rt_analysis = runtime_analysis

        self.logging = logging

    def run(self):

        time_stepping(
            self.solver,
            self.u,
            self.u0,
            self.T_final,
            self.dt,
            dt_increase=1.0,
            dt_max=1e-2,
            event_handler=self.experiment,
            output=self.output,
            runtime_analysis=self.rt_analysis,
            **self.event_params,
            logging=self.logging
        )
