import dolfinx as dfx

from mpi4py.MPI import COMM_SELF as comm

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    charge_discharge_stop,
    AnalyzeOCP,
    Simulation
    )
from pyMoBiMP.fenicsx_utils import get_mesh_spacing


def experiment(t, u, I_charge, cell_voltage=None, **kwargs):

    return charge_discharge_stop(t, u, I_charge, c_of_y=c_of_y, **kwargs)


if __name__ == "__main__":

    # Discretization
    # --------------

    # Set up the mesh
    n_elem = 32

    mesh = dfx.mesh.create_unit_interval(comm, n_elem)

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    rt_analysis = AnalyzeOCP(c_of_y=c_of_y)

    sim = Simulation(
        mesh,
        runtime_analysis=rt_analysis,
        experiment=experiment,
    )

    sim.run()
