import dolfinx

from mpi4py.MPI import COMM_WORLD as comm_world

from pathlib import Path

import numpy as np

from cahn_hilliard_utils import AnalyzeOCP
from cahn_hilliard_utils import Simulation
from cahn_hilliard_utils import _free_energy as free_energy_general
from cahn_hilliard_utils import c_of_y
from cahn_hilliard_utils import charge_discharge_stop

from gmsh_utils import dfx_spherical_mesh

from plotting_utils import PyvistaAnimation


def free_energy(c):
    return free_energy_general(c, a=0., b=0., c=0.) + \
        (2. * c - 1.) + 0.5 * (6. * c * (1. - c) - 1. / 3. * (8. * c * (1. - c) - 1) * (2 * c - 1.))


def experiment(t, u, I_charge):
    return charge_discharge_stop(t, u, I_charge, stop_on_full=True)

exp_currents = np.array([1e-2, 1e-1, 1., 2.])


if __name__ == "__main__":

    base_dir = Path("simulation_output")
    material_dir = Path("non_linear_material")
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
                         gamma=1e-3,
                         dt_fac_ini=1e-3,
                         logging=False)

        sim.run()
