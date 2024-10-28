# %%
import dolfinx as dfx

import logging

from mpi4py import MPI

import numpy as np

import os

import tqdm

from pyMoBiMP.battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DefaultPhysicalSetup,
    DFNSimulationBase as SimulationBase,
)

from pyMoBiMP.fenicsx_utils import (
    FileOutput,
    NewtonSolver,
    NonlinearProblemBlock)


log = logging.getLogger(__name__)


class Simulation(SimulationBase):
    Experiment = ChargeDischargeExperiment
    Output = FileOutput
    PhysicalSetup = DefaultPhysicalSetup
    RuntimeAnalysis = AnalyzeOCP

    def initial_data(self):

        V0, _ = self.u0.function_space.sub(0).collapse()

        eps = 1e-3

        c = dfx.fem.Function(V0)
        c.interpolate(lambda x: eps + x[1] * (1 - 2 * eps))

        y_expr = dfx.fem.Expression(
            self.y_of_c(c), V0.element.interpolation_points()
        )

        self.u0.sub(0).interpolate(y_expr)

        # Do some diagnostic checks to see whether the solver did ok.
        residual = dfx.fem.form(self.F)

        error_before = (dfx.fem.petsc.assemble_vector(residual).norm())

        # Set up a dedicated solver for the initial solve to make sure we
        # have enough iterations.
        F = [self.F, self.voltage_form]
        w = [self.u, self.voltage]

        problem = NonlinearProblemBlock(F, w)
        solver = NewtonSolver(self.mesh.comm, problem, max_iterations=50)
        self.linear_solver_setup(solver)

        its, _ = solver.solve([self.u, self.voltage])
        error = dfx.fem.petsc.assemble_vector(residual).norm()
        with tqdm.contrib.logging.logging_redirect_tqdm():
            log.info(
                f"Initial data: ini_err = {error_before}, its={its}, res = {error:1.3e}")
        assert np.isclose(error, 0.)


comm_world = MPI.COMM_WORLD


if __name__ == "__main__":

    Simulation.Experiment.c_rate = 0.

    output_destination = os.path.dirname(__file__) + "/simulation_output/equilibration"

    simulation = Simulation(
        n_particles=128,
        output_destination=output_destination
    )

    simulation.run(t_final=10.)
