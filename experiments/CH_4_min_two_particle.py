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
    charge_discharge_stop,
    y_of_c,
    c_of_y,
)

from pyMoBiMP.fenicsx_utils import (
    evaluation_points_and_cells,
    get_mesh_spacing,
    time_stepping,
    NewtonSolver,
    FileOutput as FileOutputBase,
    Fenicx1DOutput,
    RuntimeAnalysisBase,
)


# %%
class Output(Fenicx1DOutput):

    def extract_output(self, u_state, t):

        V = self.u_state.function_space

        output_snapshot = []

        for i_comp in range(V.num_sub_spaces):

            V_sub = V.sub(i_comp)

            for k_comp in range(V_sub.num_sub_spaces):

                func = u_state.sub(i_comp).sub(k_comp)

                values = func.eval(self.x_eval, self.cells)

                output_snapshot.append(values)

        return output_snapshot


class FileOutput(FileOutputBase):

    def extract_output(self, u_state, t):
        V = self.u_state.function_space

        num_vars = V.num_sub_spaces

        ret = []

        for i in range(num_vars):

            V_sub, _ = V.sub(i).collapse()

            if i == 0:
                name = "y"
            elif i == 1:
                name = "mu"
            else:
                raise ValueError(f"No component with index {i} available!")

            num_comp = V_sub.num_sub_spaces

            for j in range(num_comp):

                func = self.u_state.sub(i).sub(j)
                func.name = name + f"_{j}"

                ret.append(func)

        return ret


class AnalyzeCellPotential(RuntimeAnalysisBase):

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

        ys, mus = u_state.split()

        cs = [self.c_of_y(y) for y in ys.split()]

        # select one reference particle
        c = cs[0]
        mu = mus[0]

        coords = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(sum([co**2 for co in coords]))

        c = ufl.variable(c)
        dFdc = ufl.diff(self.free_energy(c), c)

        charge = sum(
            [dfx.fem.assemble_scalar(
                dfx.fem.form(3 * c * r**2 * ufl.dx)) for c in cs])

        chem_pot = dfx.fem.form(3 * dFdc * r**2 * ufl.dx)
        chem_pot = dfx.fem.assemble_scalar(chem_pot)

        mu_bc = dfx.fem.form(mu * r**2 * ufl.ds)
        mu_bc = dfx.fem.assemble_scalar(mu_bc)

        self.data.append([charge, chem_pot, mu_bc])

        if self.filename is not None:
            with open(self.filename, "a") as file:
                np.savetxt(file, np.array([[t, charge, chem_pot, mu_bc]]))

        return super().analyze(u_state, t)


comm_world = MPI.COMM_WORLD


if __name__ == "__main__":

    # %%
    # Discretization
    # --------------

    # Set up the mesh
    n_elem = 128

    mesh_filename = "Meshes/line_mesh.xdmf"

    if os.path.isfile(mesh_filename):
        # Load mesh from file
        with dfx.io.XDMFFile(comm_world, mesh_filename, 'r') as file:
            mesh = file.read_mesh(name="Grid")
    else:
        mesh = dfx.mesh.create_unit_interval(comm_world, n_elem)

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    # For later plotting use
    x = np.linspace(0, 1, 101)
    points_on_proc, cells = evaluation_points_and_cells(mesh, x)

    # Initial timestep size
    dt = dfx.fem.Constant(mesh, dx_cell * 0.01)

    # %%
    elem1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    elem_c = elem1
    elem_mu = elem1

    num_particles = 2

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

    # %%
    # The mixed-element functions
    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    # %%
    # Compute the chemical potential df/dc
    a = 6.0 / 4
    b = 0.2
    cc = 5

    def free_energy(u, log, sin):
        return (
            u * log(u)
            + (1 - u) * log(1 - u)
            + a * u * (1 - u)
            + b * sin(cc * np.pi * u)
        )

    eps = 1e-2

    res_c_left = sp.optimize.minimize_scalar(
        lambda c: free_energy(c, np.log, np.sin),
        bracket=(2 * eps, 0.05),
        bounds=(eps, 0.05))

    assert res_c_left.success
    c_left = res_c_left.x

    res_c_right = sp.optimize.minimize_scalar(
        lambda c: free_energy(c, np.log, np.sin),
        bracket=(0.95, 1 - 2 * eps),
        bounds=(0.95, 1 - eps))

    assert res_c_right.success
    c_right = res_c_right.x

    # %%
    # Experimental setup
    # ------------------

    # charging current
    I_charge = dfx.fem.Constant(mesh, 1e-1)

    T_final = 2.0 / I_charge.value if I_charge.value > 0 else 2.0  # ending time

    def experiment(t, u, I_charge, **kwargs):

        return charge_discharge_stop(t, u, I_charge, c_of_y=c_of_y, **kwargs)

    event_params = dict(
        I_charge=I_charge,
        stop_on_full=False,
        stop_at_empty=False,
        cycling=False,
        logging=True,
    )

    # %%
    # The variational form
    # --------------------

    form_weights = None

    def M(c):
        return c * (1 - c)

    alpha = 100.0
    theta = 1.0

    h = ufl.Circumradius(mesh)

    y, mu = ufl.split(u)
    y0, mu0 = ufl.split(u0)

    y1, y2 = ufl.split(y)
    y01, y02 = ufl.split(y0)

    mu1, mu2 = ufl.split(mu)
    mu01, mu02 = ufl.split(mu0)

    c1, c2 = c_of_y(y1), c_of_y(y2)

    v_c, v_mu = ufl.TestFunctions(V)

    v_c1, v_c2 = ufl.split(v_c)
    v_mu1, v_mu2 = ufl.split(v_mu)

    mu_theta1, mu_theta2 = (
        theta * mu1 + (theta - 1.0) * mu01,
        theta * mu2 + (theta - 1.0) * mu02,
    )

    # particle parameters
    R1 = 1.
    R2 = 1.

    A1 = 4 * np.pi * R1
    A2 = 4 * np.pi * R2

    A = A1 + A2

    # fraction of the total surface
    a1 = A1 / A
    a2 = A2 / A

    # Coupling parameters between particle surface potential.
    L1, L2 = 1.e1, 1.0e1
    L = a1 * L1 + a2 * L2

    # I * (A_1 + A_2) = I_1 * A_1 + I_2 * A_2
    term = L1 * a1 * mu_theta1 + L2 * a2 * mu_theta2

    # Here must ne a negative sign since with the I_charges, we measure
    # what flows out of the particle.
    Voltage = - I_charge / L - term / L

    # TODO: Check the sign! Somehow, there must be a minus for the
    # code to work. I think, I_charge as constructed here is the current
    # OUT OF the particle.
    I_charge1 = - L1 * (mu_theta1 + Voltage) / A1
    I_charge2 = - L2 * (mu_theta2 + Voltage) / A2

    F1 = cahn_hilliard_form(
        mesh,
        (y1, mu1),
        (y01, mu01),
        (v_c1, v_mu1),
        dt,
        M=M,
        c_of_y=c_of_y,
        free_energy=lambda c: free_energy(c, ufl.ln, ufl.sin),
        theta=theta,
        lam=0.1,
        I_charge=I_charge1,
    )

    F2 = cahn_hilliard_form(
        mesh,
        (y2, mu2),
        (y02, mu02),
        (v_c2, v_mu2),
        dt,
        M=M,
        c_of_y=c_of_y,
        free_energy=lambda c: free_energy(c, ufl.ln, ufl.sin),
        theta=theta,
        lam=0.1,
        I_charge=I_charge2,
    )

    F = F1 + F2

    # %%
    # boundary conditions
    # -------------------

    def boundary_locator(x):
        return np.isclose(x[0], 1)

    # facets
    tdim = mesh.topology.dim - 1

    facets = dfx.mesh.locate_entities_boundary(mesh, tdim, boundary_locator)
    dofs1 = dfx.fem.locate_dofs_topological(
        (V.sub(1).sub(0), V.sub(1).sub(1)), tdim, facets
    )
    dofs2 = dfx.fem.locate_dofs_topological(
        (V.sub(1).sub(1), V.sub(1).sub(0)), tdim, facets
    )

    bcs = [
        dfx.fem.dirichletbc(u.sub(1).sub(1), dofs1, V.sub(1).sub(0)),
        dfx.fem.dirichletbc(u.sub(1).sub(0), dofs2, V.sub(1).sub(1)),
    ]

    # %%
    # Initial data
    # ------------

    u_ini = dfx.fem.Function(V)

    # Constant
    def c_ini_fun(x):
        return eps * np.ones_like(x[0])

    # Store concentration-like quantity into state vector
    # ---------------------------------------------------

    V_c, _ = V.sub(0).collapse()

    c_ini = dfx.fem.Function(V_c)

    c_ini.sub(0).interpolate(lambda x: c_left + 0 * c_ini_fun(x))
    c_ini.sub(1).interpolate(lambda x: c_right - 0 * c_ini_fun(x))

    y_ini1 = dfx.fem.Expression(
        y_of_c(c_ini.sub(0)),
        c_ini.sub(0).function_space.element.interpolation_points()
    )
    y_ini2 = dfx.fem.Expression(
        y_of_c(c_ini.sub(1)),
        c_ini.sub(1).function_space.element.interpolation_points()
    )

    u_ini.sub(0).sub(0).interpolate(y_ini1)
    u_ini.sub(0).sub(1).interpolate(y_ini2)

    # Store chemical potential into state vector
    # ------------------------------------------

    c_ini1 = ufl.variable(c_ini.sub(0))
    dFdc1 = ufl.diff(free_energy(c_ini1, ufl.ln, ufl.sin), c_ini1)

    W = u_ini.sub(1).sub(0).function_space
    u_ini.sub(1).sub(0).interpolate(
        dfx.fem.Expression(dFdc1, W.element.interpolation_points())
    )

    c_ini2 = ufl.variable(c_ini.sub(1))
    dFdc2 = ufl.diff(free_energy(c_ini2, ufl.ln, ufl.sin), c_ini2)

    W = u_ini.sub(1).sub(1).function_space
    u_ini.sub(1).sub(1).interpolate(
        dfx.fem.Expression(dFdc2, W.element.interpolation_points())
    )

    u_ini.x.scatter_forward()

    # %%
    problem = NonlinearProblem(F, u)

    solver = NewtonSolver(comm_world, problem)

    # %%
    # Set up experiment
    # -----------------

    u.interpolate(u_ini)

    n_out = 501

    output_np = Output(u, np.linspace(0, T_final, n_out), x)

    results_folder = Path("simulation_output")
    results_folder.mkdir(exist_ok=True, parents=True)

    base_filename = "CH_4_min_2_particle"

    filename = results_folder / (base_filename + ".xdmf")

    output_xdmf = FileOutput(u, np.linspace(0, T_final, 51), filename=filename)

    rt_analysis = AnalyzeCellPotential(
        c_of_y=c_of_y, filename=results_folder / (base_filename + "_rt.txt")
    )

    # %%
    # Run the experiment
    # ------------------

    time_stepping(
        solver,
        u,
        u0,
        T_final,
        dt,
        dt_max=1e-3,
        dt_min=1e-12,
        tol=1e-7,
        event_handler=lambda *args, **kwargs: False,
        output=(output_np, output_xdmf),
        runtime_analysis=rt_analysis,
        **event_params,
    )
