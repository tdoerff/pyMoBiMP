import os

from petsc4py import PETSc

from pyMoBiMP.dfn_battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DefaultPhysicalSetup,
    DFNSimulationBase,
    Timer)

from pyMoBiMP.fenicsx_utils import FileOutput


class Simulation(DFNSimulationBase):
    PhysicalSetup = DefaultPhysicalSetup
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment

    def linear_solver_setup(self, solver):

        ksp = solver.ksp
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")


if __name__ == "__main__":

    simulation = Simulation(
        n_particles=1024,
        n_radius=1024,
        output_destination="_2_b_deleted",
        max_iterations=5,
    )

    comm = simulation.comm

    with Timer(f"Running on {comm.size} procs") as timer:
        simulation.run(dt_max=1e-2, n_out=0, t_final=1e-8)

    # Make sure the output file ends up in the same dir.
    output_file = os.path.dirname(__file__) + "/dfn_scaling.csv"

    if simulation.comm.rank == 0:
        with open(output_file, 'a') as file:
            file.write(f"{comm.size} {timer.elapsed_time}\n")
