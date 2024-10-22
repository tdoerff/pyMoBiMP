import dolfinx as dfx

from mpi4py.MPI import COMM_WORLD as comm

import numpy as np

from petsc4py import PETSc

import ufl

from pyMoBiMP.fenicsx_utils import read_data

from CH_4_DFN_grid import (
    AnalyzeOCP,
    c_of_y,
    DFN_FEM_form,
    DFN_function_space,
    FileOutput,
    NewtonSolver,
    NonlinearProblem,
    plot_solution_on_grid,
    TestCurrent,
    time_stepping,
    Voltage,
)


if __name__ == "__main__":

    restart_file_name = "CH_4_DFN.xdmf"

    with dfx.io.XDMFFile(comm, restart_file_name, 'r') as file:
        mesh = file.read_mesh(name="mesh")

    num_particles, t, x_data, u_data, rt_data = read_data("CH_4_DFN.h5")

    V = DFN_function_space(mesh)

    u = dfx.fem.Function(V)
    u0 = dfx.fem.Function(V)

    v = ufl.TestFunction(V)

    # Initial data
    from pyMoBiMP.battery_model import y_of_c

    V0, _ = V.sub(0).collapse()
    c = dfx.fem.Function(V0)

    c.x.array[:] = u_data[-1, 0, :].flatten()

    y_expr = dfx.fem.Expression(y_of_c(c), V0.element.interpolation_points())

    u.sub(0).interpolate(c)
    u0.interpolate(u)

    plot_solution_on_grid(u.sub(0).collapse())

    dt_min = 1e-9
    dt_max = 1e-4

    dt = dfx.fem.Constant(mesh, 1e-8)

    T_final = 650.0

    I_global = dfx.fem.Constant(mesh, 0.01)
    V_cell = Voltage(u, I_global)

    # FEM Form
    # ========
    F = DFN_FEM_form(u, u0, v, dt, V_cell)

    # %% Runtime analysis and output
    # ==============================
    rt_analysis = AnalyzeOCP(u,
                             c_of_y,
                             V_cell,
                             filename="CH_4_DFN_restarted_rt.txt")

    output = FileOutput(u,
                        np.linspace(0, T_final, 101),
                        filename="CH_4_DFN_restarted.xdmf",
                        variable_transform=c_of_y)

    callback = TestCurrent(u, V_cell, I_global)

    # %% DOLFINx problem and solver setup
    # ===================================

    problem = NonlinearProblem(F, u, callback=callback)
    solver = NewtonSolver(comm, problem, callback=lambda solver, uh: V_cell.update())
    solver.rtol = 1e-10
    solver.max_it = 50
    solver.convergence_criterion = "incremental"
    solver.relaxation_parameter = 1.0

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "ksp"
    ksp.setFromOptions()

    dt.value = 0.

    residual = dfx.fem.form(F)

    print(dfx.fem.petsc.assemble_vector(residual).norm())
    its, success = solver.solve(u)
    error = dfx.fem.petsc.assemble_vector(residual).norm()
    print(its, error)
    assert np.isclose(error, 0.)

    # %% Run the simulation
    # =====================
    time_stepping(
        solver,
        u,
        u0,
        T_final,
        dt,
        V_cell,
        dt_max=dt_max,
        dt_min=dt_min,
        t_start=t[-1],
        dt_increase=1.1,
        tol=1e-4,
        runtime_analysis=rt_analysis,
        output=output,
        callback=callback
    )
