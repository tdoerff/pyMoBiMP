# %%
import dolfinx as dfx

from dolfinx.fem.petsc import NonlinearProblem

from mpi4py import MPI

import numpy as np

import os

from pathlib import Path

import scipy as sp

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    cahn_hilliard_form,
    y_of_c,
    c_of_y,
)

from pyMoBiMP.fenicsx_utils import (
    get_mesh_spacing,
    time_stepping,
    NewtonSolver,
    FileOutput,
    RuntimeAnalysisBase,
    strip_off_xdmf_file_ending
)


# %%
class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        Ls,
        As,
        I_charge,
        *args,
        c_of_y=c_of_y,
        free_energy=lambda u: 0.5 * u**2,
        filename=None,
        u_state=None,
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
        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):
        charge = sum([dfx.fem.assemble_scalar(
            self.charge_form[i]) for i in range(self.n)])

        chem_pot = dfx.fem.assemble_scalar(self.chem_pot_form)
        mu_bc = dfx.fem.assemble_scalar(self.mu_bc_form)

        mus_bc = [dfx.fem.assemble_scalar(self.mus_bc_form[i]) for i in range(self.n)]

        cell_voltage = self.I_charge.value / self.L + sum(
            L_ / self.L * a_ * mu_ for L_, a_, mu_ in zip(self.Ls, self.aas, mus_bc)
        )

        self.data.append([charge, chem_pot, mu_bc, cell_voltage])

        return super().analyze(u_state, t)


comm_world = MPI.COMM_WORLD


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

    def __init__(self,
                 mesh,
                 output_destination,
                 num_particles=12,
                 n_out=501,
                 C_rate=0.01):

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

        # charging current
        I_charge = dfx.fem.Constant(mesh, 1. / 3. * C_rate)

        T_final = self.T_final

        # %%
        # The variational form
        # --------------------

        def M(c):
            return c * (1 - c)

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
                c_of_y=c_of_y,
                free_energy=lambda c: self.free_energy(c, ufl.ln, ufl.sin),
                theta=theta,
                lam=0.1,
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
        problem = NonlinearProblem(F, u)

        self.solver = NewtonSolver(comm_world, problem)

        self.solver.tol = 1e-3
        # %%
        # Set up experiment
        # -----------------

        u.interpolate(u_ini)

        output_destination = strip_off_xdmf_file_ending(output_destination)

        results_file = Path(output_destination + ".xdmf")

        results_folder = Path(os.path.dirname(str(results_file)))
        results_folder.mkdir(exist_ok=True, parents=True)

        xdmf_filename = results_file
        rt_filename = output_destination + "_rt.txt"

        self.output_xdmf = FileOutput(
            u, np.linspace(0, T_final, n_out), filename=xdmf_filename)

        self.rt_analysis = AnalyzeCellPotential(
            Ls, As, I_charge,
            c_of_y=c_of_y,
            filename=rt_filename,
            u_state=u,
            num_particles=num_particles,
        )

        # Finalize with some variables that need to be attached to the class instance.
        self.u = u
        self.u0 = u0
        self.dt = dt

        coords = ufl.SpatialCoordinate(u.function_space.mesh)
        r_square = ufl.inner(coords, coords)

        y, _ = u.split()

        cs = [c_of_y(y_) for y_ in y]

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
        T = 6.0 / self.C_rate if self.C_rate > 0 else 2.0  # ending time
        return T

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
        c_of_y=c_of_y,
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


Simulation = MultiParticleSimulation


if __name__ == "__main__":

    # %%
    # Discretization
    # --------------

    # Set up the mesh
    mesh_filename = "Meshes/line_mesh.xdmf"

    if os.path.isfile(mesh_filename):
        # Load mesh from file
        with dfx.io.XDMFFile(comm_world, mesh_filename, 'r') as file:
            mesh = file.read_mesh(name="Grid")
    else:
        n_elem = 16
        mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    simulation = Simulation(
        mesh,
        output_destination="simulation_output/CH_4_min_10_particles")

    simulation.run(tol=1e-5)
