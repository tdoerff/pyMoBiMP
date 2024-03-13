import dolfinx

from mpi4py.MPI import COMM_WORLD as comm_world

from pathlib import Path

import numpy as np

from cahn_hilliard_utils import AnalyzeOCP
from cahn_hilliard_utils import Simulation
from cahn_hilliard_utils import _free_energy as free_energy_general
from cahn_hilliard_utils import c_of_y

from gmsh_utils import dfx_spherical_mesh

from plotting_utils import PyvistaAnimation


def ideal_free_energy(c):
    return free_energy_general(c, a=0., b=0., c=0.)

if __name__ == "__main__":

    base_dir = Path("simulation_output")
    material_dir = Path("ideal_material")
    exp_dir = Path("const_current")

    results_folder = base_dir / material_dir / exp_dir
    results_folder.mkdir(exist_ok=True, parents=True)

    mesh = dolfinx.mesh.create_unit_interval(comm_world, 128)

    for I in np.logspace(-2, 1, 4, base=10):

        print(f">>> I = {I:1.3e}")

        rt_analysis = AnalyzeOCP(c_of_y=c_of_y, free_energy=ideal_free_energy,
                                 filename=results_folder / f"I_{I:1.3e}.txt")

        sim = Simulation(mesh=mesh,
                        free_energy=ideal_free_energy,
                        T_final=2.,
                        output_file=None,
                        runtime_analysis=rt_analysis,
                        I=I)

        sim.run()

        ana_out_array = np.array([(t, *data) for t, data in zip(sim.rt_analysis.t, sim.rt_analysis.data)])

        # np.savetxt(results_folder / f"I_{I:1.3e}.txt", ana_out_array)

    # FIXME: The output below is a workaround due to
    # non-functional VTK/XDMF/... output.
    # mesh_3d, _, _ = dfx_spherical_mesh(comm_world, resolution=1.0)

    # anim = PyvistaAnimation(
    #     sim.output,
    #     mesh_3d=mesh_3d,
    #     c_of_y=lambda y: np.exp(y) / (1 + np.exp(y)),
    #     res=1.0,
    #     clim=[0.0, 1.0],
    #     cmap="hot",
    # )

    # anim.write_vtk_output("simulation_output/ideal_material/constant_current.vtk")
