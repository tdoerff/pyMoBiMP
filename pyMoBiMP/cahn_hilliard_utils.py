from collections.abc import Callable

import dolfinx as dfx

from dolfinx.fem.petsc import NonlinearProblem

from mpi4py import MPI

import numpy as np

import os

import pathlib

from typing import Dict, Optional, overload, Union

import ufl

from .exceptions import WrongNumberOfArguments

from .fenicsx_utils import (
    get_mesh_spacing,
    NewtonSolver,
    RuntimeAnalysisBase,
    time_stepping,
    FileOutput,
)


# Forward and backward variable transformation.
def c_of_y(y):
    return ufl.exp(y) / (1 + ufl.exp(y))


def y_of_c(c):
    return ufl.ln(c / (1 - c))


@overload
def cahn_hilliard_form(
    u: dfx.fem.Function,
    u0: dfx.fem.Function,
    dt: Union[dfx.fem.Constant, float],
    M: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 1.0,
    c_of_y: Callable[[dfx.fem.Function], dfx.fem.Expression] = c_of_y,
    free_energy: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.25
    * (c**2 - 1) ** 2,
    lam: float = 0.01,
    I_charge: Union[float, dfx.fem.Constant] = 0.1,
    grad_c_bc: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.0 * c,
    theta: Union[float, dfx.fem.Constant] = 1.0,
    form_weights: Optional[Dict[str, dfx.fem.Expression]] = None,
) -> dfx.fem.Form: ...


@overload
def cahn_hilliard_form(
    mesh: dfx.mesh.Mesh,
    u: dfx.fem.Function,
    u0: dfx.fem.Function,
    v: ufl.Coargument,
    dt: Union[dfx.fem.Constant, float],
    M: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 1.0,
    c_of_y: Callable[[dfx.fem.Function], dfx.fem.Expression] = c_of_y,
    free_energy: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.25
    * (c**2 - 1) ** 2,
    lam: float = 0.01,
    I_charge: Union[float, dfx.fem.Constant] = 0.1,
    grad_c_bc: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.0 * c,
    theta: Union[float, dfx.fem.Constant] = 1.0,
    form_weights: Optional[Dict[str, dfx.fem.Expression]] = None,
) -> dfx.fem.Form: ...


def cahn_hilliard_form(
    *args,
    M: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 1.0,
    c_of_y: Callable[[dfx.fem.Function], dfx.fem.Expression] = c_of_y,
    free_energy: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.25
    * (c**2 - 1) ** 2,
    lam: float = 0.01,
    I_charge: Union[float, dfx.fem.Constant] = 0.1,
    grad_c_bc: Callable[[dfx.fem.Function], dfx.fem.Expression] = lambda c: 0.0 * c,
    theta: Union[float, dfx.fem.Constant] = 1.0,
    form_weights: Optional[Dict[str, dfx.fem.Expression]] = None,
) -> dfx.fem.Form:
    """Compose the FEM form for the Cahn-Hilliard equation.

    Parameters
    ----------
    *args :
    M : Callable[[dfx.fem.Function], dfx.fem.Expression], optional
        diffusivity, by default lambda c: 1.
    c_of_y : Callable[[dfx.fem.Function], dfx.fem.Expression], optional
        varable transform, by default c_of_y
    free_energy : Callable[[dfx.fem.Function], dfx.fem.Expression], optional
        free energy density function, by default lambda c: 0.25*(c**2 - 1)**2
    lam : float, optional
        phase separation parameter, by default 0.01
    I_charge : Union[float, dfx.fem.Constant], optional
        current density through surface, by default 0.1
    grad_c_bc : Callable[[dfx.fem.Function], dfx.fem.Expression], optional
        Neumann condition for c at surface, by default lambda c: 0.*c
    theta : Union[float, dfx.fem.Constant], optional
        parameter controlling the Theta-timestepping scheme, by default 1.0
    form_weights : Dict[str, dfx.fem.Expression], optional
        surface and volume element weights, by default O=None, leads to spherical geometry

    Returns
    -------
    dfx.fem.Form
        The assembled single-domain (i.e., single-particle) Cahn-Hilliard form

    Raises
    ------
    WrongNumberOfArguments
        *args must contain 3 or 5 elements.
    """

    # Timestep is supposed to be the last positional argument
    dt = args[-1]

    if len(args) == 3:

        psi = args[0]
        psi0 = args[1]

        assert psi.function_space is psi0.function_space

        V = psi.function_space
        mesh = V.mesh

        # Split the functions
        y, mu = ufl.split(psi)
        y0, mu0 = ufl.split(psi0)

        v_c, v_mu = ufl.TestFunctions(V)

    elif len(args) == 5:
        mesh = args[0]

        y, mu = args[1]
        y0, mu0 = args[2]

        v_c, v_mu = args[3]

    else:
        raise WrongNumberOfArguments(
            "cahn_hilliard_form takes either 3 or 5 positional arguments."
        )

    y = ufl.variable(y)
    c = c_of_y(y)

    dcdy = ufl.diff(c, y)

    # Differentiate the free energy function to
    # obtain the chemical potential
    c = ufl.variable(c)
    dfdc = ufl.diff(free_energy(c), c)
    mu_chem = dfdc

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
    F1 -= I_charge * s_A * v_c * dt * ds

    F2 = s_V * mu * v_mu * dx
    F2 -= s_V * mu_chem * v_mu * dx
    F2 -= lam * (s_V * ufl.inner(ufl.grad(c), ufl.grad(v_mu)) * dx)
    F2 += grad_c_bc(c) * (s_A * v_mu * ds)

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
    stop_on_full=True,
    cycling=True,
    logging=False,
):

    coords = ufl.SpatialCoordinate(u.function_space.mesh)
    r = ufl.sqrt(sum([c**2 for c in coords]))

    y, _ = u.split()

    c = c_of_y(y)

    # This is a bit hackish, since we just need to multiply by a function that
    # is zero at r=0 and 1 at r=1.
    c_bc = dfx.fem.form(r**2 * c * ufl.ds)
    c_bc = dfx.fem.assemble_scalar(c_bc)

    if logging:
        print(f"t={t:1.5f} ; c_bc = {c_bc:1.3e}", c_bounds)

    if c_bc > c_bounds[1] and I_charge.value > 0.0:
        print(
            ">>> charge at boundary exceeds maximum " +
            f"(max(c) = {c_bc:1.3f} > {c_bounds[1]:1.3f})."
        )

        if stop_on_full:
            print(">>> Particle is filled.")

            return True

        print(">>> Start discharging.")
        I_charge.value *= -1.0

        return False

    if c_bc < c_bounds[0] and I_charge.value < 0.0:

        if stop_at_empty:
            print(">>> Particle is emptied!")

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

        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):

        V = u_state.function_space
        mesh = V.mesh

        y, mu = u_state.split()

        c = self.c_of_y(y)

        coords = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(sum([co**2 for co in coords]))

        c = ufl.variable(c)
        dFdc = ufl.diff(self.free_energy(c), c)

        charge = dfx.fem.form(3 * c * r**2 * ufl.dx)
        charge = dfx.fem.assemble_scalar(charge)

        chem_pot = dfx.fem.form(3 * dFdc * r**2 * ufl.dx)
        chem_pot = dfx.fem.assemble_scalar(chem_pot)

        mu_bc = dfx.fem.form(mu * r**2 * ufl.ds)
        mu_bc = dfx.fem.assemble_scalar(mu_bc)

        self.data.append([charge, chem_pot, mu_bc])

        return super().analyze(u_state, t)


# Defaults for the Simulation class
_mesh = dfx.mesh.create_unit_interval(MPI.COMM_WORLD, 128)


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
        ] = lambda c: 1.0
        * c
        * (1 - c),
        I: dfx.fem.Constant | float = 1.0,
        eps: float = 1e-3,
        dt_fac_ini: float = 1e-2,
        c_ini=lambda x, eps: eps * np.ones_like(x[0]),
        output_file: Optional[str | os.PathLike] = None,
        n_out: int = 51,
        runtime_analysis: Optional[RuntimeAnalysisBase] = None,
        logging: bool = True,
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

        # Experimental setup
        # ------------------

        # Charging current goes into the form and is recomputed in experiment
        if isinstance(I, float):
            I_charge = dfx.fem.Constant(mesh, I)
        elif isinstance(I, dfx.fem.Constant):
            I_charge = I
        else:
            raise TypeError(f"I of unsupported type {type(I)}")

        self.event_params = dict(I_charge=I_charge)

        self.experiment = experiment

        # (initial) timestep
        # ------------------
        dx = get_mesh_spacing(mesh)
        self.dt = dfx.fem.Constant(mesh, dt_fac_ini * dx / I_charge.value)

        self.T_final = T_final

        # The mixed-element functions
        # ---------------------------
        self.u = dfx.fem.Function(V)
        self.u0 = dfx.fem.Function(V)

        # The weak form of the problem
        # ----------------------------
        F = cahn_hilliard_form(
            self.u,
            self.u0,
            self.dt,
            free_energy=free_energy,
            theta=0.75,
            c_of_y=lambda y: c_of_y(y),
            M=M,
            lam=gamma,
            **self.event_params,
        )

        # Initial data
        # ------------

        populate_initial_data(self.u, lambda r: c_ini(r, eps), free_energy)

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
            self.output = FileOutput(
                self.u,
                np.linspace(0, T_final, n_out),
                output_file,
                variable_transform=c_of_y,
            )
        else:
            self.output = None

        self.rt_analysis = runtime_analysis

        self.logging = logging

    def run(self, *args, dt_increase=1.0, dt_max=1e-2, **kwargs):

        time_stepping(
            self.solver,
            self.u,
            self.u0,
            self.T_final,
            self.dt,
            dt_increase=dt_increase,
            dt_max=dt_max,
            event_handler=self.experiment,
            output=self.output,
            runtime_analysis=self.rt_analysis,
            **self.event_params,
            **kwargs,
            logging=self.logging,
        )
