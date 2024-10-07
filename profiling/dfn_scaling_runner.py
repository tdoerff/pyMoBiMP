import time

import os

from pyMoBiMP.dfn_battery_model import (
    AnalyzeOCP,
    ChargeDischargeExperiment,
    DFNSimulationBase)

from pyMoBiMP.fenicsx_utils import FileOutput


class Simulation(DFNSimulationBase):
    Output = FileOutput
    RuntimeAnalysis = AnalyzeOCP
    Experiment = ChargeDischargeExperiment


def mpi_time():

    # Use the barriers to make sure every process starts and
    # finishs at the same time and finishes
    comm.barrier()
    tic = time.perf_counter()

    return tic


if __name__ == "__main__":

    simulation = Simulation(
        n_particles=1024,
        n_radius=16,
        output_destination="_2_b_deleted"
    )

    comm = simulation.comm

    tic = mpi_time()

    simulation.run(dt_max=1e-4, n_out=0, t_final=0.01,)

    toc = mpi_time()

    # elapsed time
    etime = toc - tic

    # Make sure the output file ends up in the same dir.
    output_file = os.path.dirname(__file__) + "/dfn_scaling.csv"

    if simulation.comm.rank == 0:
        with open(output_file, 'a') as file:
            file.write(f"{comm.size} {etime}\n")
