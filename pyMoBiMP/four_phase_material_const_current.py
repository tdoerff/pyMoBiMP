import dolfinx

from mpi4py.MPI import COMM_WORLD as comm_world

from pathlib import Path

import numpy as np

from cahn_hilliard_utils import AnalyzeOCP
from cahn_hilliard_utils import Simulation
from cahn_hilliard_utils import _free_energy as free_energy_general
from cahn_hilliard_utils import c_of_y
from cahn_hilliard_utils import charge_discharge_stop

from ideal_material_const_current import exp_currents


def free_energy(c):
    return free_energy_general(c)


def experiment(t, u, I_charge):
    return charge_discharge_stop(t, u, I_charge, stop_on_full=True)


if __name__ == "__main__":

    base_dir = Path("simulation_output")
    material_dir = Path("four_phase_material")
    exp_dir = Path("const_current")

    results_folder = base_dir / material_dir / exp_dir
    results_folder.mkdir(exist_ok=True, parents=True)

    mesh = dolfinx.mesh.create_unit_interval(comm_world, 128)

    for I in exp_currents:

        print(f">>> I = {I:1.3e}")

        rt_analysis = AnalyzeOCP(c_of_y=c_of_y, free_energy=free_energy,
                                 filename=results_folder / f"I_{I:1.3e}.txt")

        sim = Simulation(mesh=mesh,
                         free_energy=free_energy,
                         T_final=70.,
                         experiment=experiment,
                         output_file=None,
                         runtime_analysis=rt_analysis,
                         I=I,
                         dt_fac_ini=1e-3,
                         logging=False)

        sim.run()
