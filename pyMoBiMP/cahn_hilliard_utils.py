from collections.abc import Callable

import dolfinx as dfx

from mpi4py import MPI

import numpy as np

import os

import pathlib

import scipy as sp

from typing import Dict, Optional, overload, Union

import ufl

from .exceptions import WrongNumberOfArguments

from .fenicsx_utils import (
    get_mesh_spacing,
    NewtonSolver,
    NonlinearProblem,
    RuntimeAnalysisBase,
    time_stepping,
    FileOutput,
    strip_off_xdmf_file_ending
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

    F1 = s_V * (c_of_y(y) - c_of_y(y0)) * v_c * dx
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
    c_bc_form,
    c_bounds=[0.05, 0.99],
    c_of_y=c_of_y,
    stop_at_empty=True,
    stop_on_full=True,
    cycling=True,
    logging=False,
):

    # This is a bit hackish, since we just need to multiply by a function that
    # is zero at r=0 and 1 at r=1.
    c_bc = dfx.fem.assemble_scalar(c_bc_form)

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

    def analyze(self, t):

        V = self.u_state.function_space
        mesh = V.mesh

        y, mu = self.u_state.split()

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

        return super().analyze(t)


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


class SingleParticleSimulation:

    NewtonSolver = NewtonSolver
    NonlinearProblem = NonlinearProblem

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
        problem = self.NonlinearProblem(F, self.u)

        self.solver = self.NewtonSolver(mesh.comm, problem)

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


class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        u_state,
        Ls,
        As,
        I_charge,
        *args,
        c_of_y=c_of_y,
        free_energy=lambda u: 0.5 * u**2,
        filename=None,
        num_particles=None,
        **kwargs,
    ):
        self.free_energy = free_energy
        self.c_of_y = c_of_y

        self.filename = filename

        self.Ls = Ls

        A = sum(As)
        self.aas = As / A
        self.I_charge = I_charge

        self.L = sum(Ls * self.aas)

        self.n = num_particles

        V = u_state.function_space
        mesh = V.mesh

        ys, mus = u_state.split()

        cs = [self.c_of_y(y) for y in ys.split()]

        # select one reference particle
        c = cs[0]
        mu = mus[0]

        coords = ufl.SpatialCoordinate(mesh)
        r_square = ufl.inner(coords, coords)

        c = ufl.variable(c)
        dFdc = ufl.diff(self.free_energy(c), c)

        self.chem_pot_form = dfx.fem.form(3 * dFdc * r_square * ufl.dx)
        self.mu_bc_form = dfx.fem.form(mu * r_square * ufl.ds)
        self.charge_form = [dfx.fem.form(3 * c * r_square * ufl.dx) for c in cs]
        self.mus_bc_form = [dfx.fem.form(mu_ * r_square * ufl.ds) for mu_ in mus]

        return super().setup(u_state, *args, **kwargs)

    def analyze(self, t):
        charge = sum([dfx.fem.assemble_scalar(
            self.charge_form[i]) for i in range(self.n)])

        chem_pot = dfx.fem.assemble_scalar(self.chem_pot_form)
        mu_bc = dfx.fem.assemble_scalar(self.mu_bc_form)

        mus_bc = [dfx.fem.assemble_scalar(self.mus_bc_form[i]) for i in range(self.n)]

        cell_voltage = self.I_charge.value / self.L + sum(
            L_ / self.L * a_ * mu_ for L_, a_, mu_ in zip(self.Ls, self.aas, mus_bc)
        )

        self.data.append([charge, chem_pot, mu_bc, cell_voltage])

        return super().analyze(t)


def compute_particle_current_densities(mus, As, Ls, I_charge):

    A = sum(As)

    # fraction of the total surface
    a_ratios = As / A

    # Coupling parameters between particle surface potential.
    L = sum([a_ * L_ for a_, L_ in zip(a_ratios, Ls)])

    # I * (A_1 + A_2) = I_1 * A_1 + I_2 * A_2
    term = sum([L_ * a_ * mu_ for L_, a_, mu_ in zip(Ls, a_ratios, mus)])

    # Here must be a negative sign since with the I_charges, we measure
    # what flows out of the particle.
    Voltage = - I_charge / L - term / L

    # TODO: Check the sign! Somehow, there must be a minus for the
    # code to work. I think, I_charge as constructed here is the current
    # OUT OF the particle.
    I_charges = [
        -L_ * (mu_ + Voltage) for L_, mu_ in zip(Ls, mus)]

    return I_charges


class MultiParticleSimulation():

    NewtonSolver = NewtonSolver
    NonlinearProblem = NonlinearProblem

    def __init__(self,
                 mesh,
                 output_destination,
                 num_particles=12,
                 n_out=501,
                 C_rate=0.01,
                 M=lambda c: c * (1 - c),
                 gamma=0.1,
                 c_of_y=c_of_y,
                 comm=MPI.COMM_WORLD):

        self.mesh = mesh
        self.C_rate = C_rate

        dx_cell = get_mesh_spacing(mesh)

        print(f"Cell spacing: h = {dx_cell}")

        # Initial timestep size
        dt = dfx.fem.Constant(mesh, dx_cell * 0.01)

        # Function space.
        self.V = self.create_function_space(mesh, num_particles)

        # The mixed-element functions.
        u = dfx.fem.Function(self.V)
        u0 = dfx.fem.Function(self.V)

        # %%
        # Experimental setup
        # ------------------
        self.c_of_y = c_of_y

        # charging current
        I_charge = dfx.fem.Constant(mesh, 1. / 3. * C_rate)

        T_final = self.T_final

        # %%
        # The variational form
        # --------------------

        theta = 1.0

        y, mu = ufl.split(u)
        y0, mu0 = ufl.split(u0)

        y1s = ufl.split(y)
        y0s = ufl.split(y0)

        mu1s = ufl.split(mu)
        mu0s = ufl.split(mu0)

        v_c, v_mu = ufl.TestFunctions(self.V)

        v_cs = ufl.split(v_c)
        v_mus = ufl.split(v_mu)

        mu_theta = [theta * mu1s_ + (theta - 1.0) * mu0s_
                    for mu1s_, mu0s_ in zip(mu1s, mu0s)]

        # particle parameters
        Rs = np.ones(num_particles)

        As = 4 * np.pi * Rs

        Ls = 1.e1 * (1 + 0.1 * (2 * np.random.random(num_particles) - 1))

        I_charges = compute_particle_current_densities(
            mu_theta, As, Ls, I_charge
        )

        # Assemble the individual particle forms.
        Fs = [
            cahn_hilliard_form(
                mesh,
                (y1_, mu1_),
                (y0_, mu0_),
                (v_c_, v_mu_),
                dt,
                M=M,
                c_of_y=self.c_of_y,
                free_energy=lambda c: self.free_energy(c, ufl.ln, ufl.sin),
                theta=theta,
                lam=gamma,
                I_charge=I_charge_,
            ) for y1_, mu1_, y0_, mu0_, v_c_, v_mu_, I_charge_ in zip(
                y1s, mu1s, y0s, mu0s, v_cs, v_mus, I_charges
            )
            ]

        # Compose the global FEM form.
        F = sum(Fs)

        # Initial data
        u_ini = self.initial_data()

        # %%
        problem = self.NonlinearProblem(F, u)

        self.solver = self.NewtonSolver(comm, problem)

        self.solver.tol = 1e-3
        # %%
        # Set up experiment
        # -----------------

        u.interpolate(u_ini)

        output_destination = strip_off_xdmf_file_ending(output_destination)

        results_file = pathlib.Path(output_destination + ".xdmf")

        results_folder = pathlib.Path(os.path.dirname(str(results_file)))
        results_folder.mkdir(exist_ok=True, parents=True)

        xdmf_filename = results_file
        rt_filename = output_destination + "_rt.txt"

        self.output_xdmf = FileOutput(
            u, np.linspace(0, T_final, n_out), filename=xdmf_filename)

        self.rt_analysis = AnalyzeCellPotential(
            u, Ls, As, I_charge,
            c_of_y=self.c_of_y,
            filename=rt_filename,
            num_particles=num_particles,
        )

        # Finalize with some variables that need to be attached to the class instance.
        self.u = u
        self.u0 = u0
        self.dt = dt

        coords = ufl.SpatialCoordinate(u.function_space.mesh)
        r_square = ufl.inner(coords, coords)

        y, _ = u.split()

        cs = [self.c_of_y(y_) for y_ in y]

        # This is a bit hackish, since we just need to multiply by a function that
        # is zero at r=0 and 1 at r=1.
        cs_bc_form = [dfx.fem.form(r_square * c_ * ufl.ds) for c_ in cs]

        self.event_params = dict(
            I_charge=I_charge,
            stop_on_full=False,
            stop_at_empty=False,
            cycling=False,
            logging=True,
            num_particles=num_particles,
            cs_bc_form=cs_bc_form
        )

    def run(self,
            dt_max=1e-1,
            dt_min=1e-8,
            tol=1e-4):
        # %%
        # Run the experiment
        # ------------------

        time_stepping(
            self.solver,
            self.u,
            self.u0,
            self.T_final,
            self.dt,
            dt_max=dt_max,
            dt_min=dt_min,
            tol=tol,
            event_handler=self.experiment,
            output=self.output_xdmf,
            runtime_analysis=self.rt_analysis,
            **self.event_params,
        )

    def create_function_space(self, mesh, num_particles):
        elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        elem_c = elem1
        elem_mu = elem1

        multi_particle_element = ufl.MixedElement(
            [
                [
                    elem_c,
                ]
                * num_particles,
                [
                    elem_mu,
                ]
                * num_particles,
            ]
        )

        V = dfx.fem.FunctionSpace(mesh, multi_particle_element)
        return V

    def initial_data(self):
        # Balanced state for initial data.
        eps = 1e-2

        res_c_left = sp.optimize.minimize_scalar(
            lambda c: self.free_energy(c, np.log, np.sin),
            bracket=(2 * eps, 0.05),
            bounds=(eps, 0.05))

        assert res_c_left.success
        c_left = res_c_left.x

        u_ini = dfx.fem.Function(self.V)

        # Constant
        def c_ini_fun(x):
            return eps * np.ones_like(x[0])

        # Store concentration-like quantity into state vector
        # ---------------------------------------------------

        V_c, _ = self.V.sub(0).collapse()

        c_ini = dfx.fem.Function(V_c)

        # extract number of particles
        y, _ = u_ini.split()

        num_particles = len(y.split())

        for i_particle in range(num_particles):

            c_ini.sub(i_particle).interpolate(lambda x: c_left + 0 * c_ini_fun(x))

            W = c_ini.sub(i_particle).function_space
            x_interpolate = W.element.interpolation_points()

            y_ini = dfx.fem.Expression(
                y_of_c(c_ini.sub(i_particle)), x_interpolate)

            u_ini.sub(0).sub(i_particle).interpolate(y_ini)

        # Store chemical potential into state vector
        # ------------------------------------------

        for i_particle in range(num_particles):
            c_ini_ = ufl.variable(c_ini.sub(i_particle))
            dFdc1 = ufl.diff(self.free_energy(c_ini_, ufl.ln, ufl.sin), c_ini_)

            W = u_ini.sub(1).sub(i_particle).function_space
            u_ini.sub(1).sub(i_particle).interpolate(
                dfx.fem.Expression(dFdc1, W.element.interpolation_points())
            )

        u_ini.x.scatter_forward()

        return u_ini

    @property
    def T_final(self):
        self._T = 6.0 / self.C_rate if self.C_rate > 0 else 2.0  # ending time
        return self._T

    @staticmethod
    def free_energy(u, log, sin):
        a = 6.0 / 4
        b = 0.2
        cc = 5

        return (
            u * log(u)
            + (1 - u) * log(1 - u)
            + a * u * (1 - u)
            + b * sin(cc * np.pi * u)
        )

    @staticmethod
    def experiment(
        t,
        u,
        I_charge,
        c_bounds=[-3.7, 3.7],
        cell_voltage=None,
        stop_at_empty=True,
        stop_on_full=True,
        cycling=True,
        logging=False,
        num_particles=None,
        cs_bc_form=None
    ):

        cs_bc = [dfx.fem.assemble_scalar(cs_bc_form[i]) for i in range(num_particles)]

        if logging:
            print(f"t={t:1.5f} ; c_bc = [{min(cs_bc):1.3e}, {max(cs_bc):1.3e}]", c_bounds)

        # Whenever you may ask yourself whether this works, mind the sign!
        # cell_voltage is the voltage computed by AnalyzeCellPotential, ie,
        # it increases with chemical potential at the surface of the particles.
        # The actual cell voltage as measured is the negative of it.
        if cell_voltage > c_bounds[1] and I_charge.value > 0.0:
            print(
                ">>> charge at boundary exceeds maximum " +
                f"(max(c) = {max(cs_bc):1.3f} > {c_bounds[1]:1.3f})."
            )

            if stop_on_full:
                print(">>> Particle is filled.")

                return True

            print(">>> Start discharging.")
            I_charge.value *= -1.0

            return False

        if cell_voltage < c_bounds[0] and I_charge.value < 0.0:

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
