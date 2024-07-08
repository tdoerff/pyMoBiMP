import dolfinx as dfx

from mpi4py.MPI import SUM, COMM_SELF as comm, COMM_WORLD as comm_world

import numpy as np

import random

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    c_of_y,
    charge_discharge_stop,
    _free_energy as free_energy,
    AnalyzeOCP,
    Simulation
    )

from pyMoBiMP.fenicsx_utils import (
    FileOutput,
    get_mesh_spacing,
    RuntimeAnalysisBase
    )


def experiment(t, u, I_charge, cell_voltage=None, **kwargs):

    return charge_discharge_stop(t, u, I_charge, c_of_y=c_of_y, **kwargs)


class AnalyzeCellPotential(RuntimeAnalysisBase):

    def setup(
        self,
        comm,
        L,
        A,
        I_charge,
        *args,
        c_of_y,
        free_energy=free_energy,
        filename=None,
        **kwargs
    ):
        self.comm = comm

        self.free_energy = free_energy
        self.c_of_y = c_of_y

        self.filename = filename

        self.L_k = L
        self.A_k = A

        self.I_charge = I_charge

        self.A = comm.allreduce(self.A_k, op=SUM)

        self.a_k = self.A_k / A

        self.L = comm.allreduce(self.L_k * self.a_k, op=SUM)

        return super().setup(*args, **kwargs)

    def analyze(self, u_state, t):

        V = u_state.function_space
        mesh = V.mesh

        y, mu = u_state.split()

        c = self.c_of_y(y)

        # TODO: this can be done at initialization.
        coords = ufl.SpatialCoordinate(mesh)
        r = ufl.sqrt(sum([co**2 for co in coords]))

        charge_k = dfx.fem.assemble_scalar(dfx.fem.form(3 * c * r**2 * ufl.dx))

        charge = self.comm.allreduce(charge_k, op=SUM)

        mu_bc = dfx.fem.assemble_scalar(dfx.fem.form(mu * r**2 * ufl.ds))

        particle_voltage = self.L_k / self.L * self.a_k * mu_bc

        cell_voltage = self.comm.allreduce(particle_voltage, op=SUM)
        cell_voltage += self.I_charge.value / self.L

        self.data.append([charge, cell_voltage])

        return super().analyze(u_state, t)


if __name__ == "__main__":

    # Discretization
    # --------------

    # Set up the mesh
    n_elem = 32

    mesh = dfx.mesh.create_unit_interval(comm, n_elem)

    dx_cell = get_mesh_spacing(mesh)

    print(f"Cell spacing: h = {dx_cell}")

    R = 1.
    A = 4 * np.pi * R**2

    I_charge = dfx.fem.Constant(mesh, 0.1)

    rt_analysis = AnalyzeCellPotential(
        comm_world,
        L=1.e1 * (1 + 0.1 * (2 * random.random() - 1)),
        A=A,
        I_charge=I_charge,
        c_of_y=c_of_y)

    sim = Simulation(
        mesh,
        runtime_analysis=rt_analysis,
        experiment=experiment,
        I=I_charge,
        # output_file=f"simulation_output/CH_4_part{comm.rank:04}.xdmf"
    )

    sim.run()
