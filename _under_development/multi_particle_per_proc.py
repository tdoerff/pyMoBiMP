# %%
import dolfinx as dfx

from mpi4py import MPI

from math import log10, ceil

import numpy as np

import os

import ufl

from pyMoBiMP.cahn_hilliard_utils import (
    MultiParticleSimulation as SimulationBase,
)

from pyMoBiMP.fenicsx_utils import StopEvent


def log(*msg, comm=MPI.COMM_WORLD, cond=True):

    if cond:
        print(f"LOG [{comm.rank}]: ", *msg, flush=True)


class Simulation(SimulationBase):

    def __init__(self, *args, logging=True, **kwargs):

        self.logging = logging

        super().__init__(*args, **kwargs)

        self.I_battery = dfx.fem.Constant(mesh, 0.1)

        # Invoke the experiment
        self.experiment = self.Experiment(self.u, self.I_battery, c_of_y=self.c_of_y)

    def run(self,
            dt_max=1e-1,
            dt_min=1e-8,
            tol=1e-4,
            dt_increase=1.1):

        dt = self.dt

        assert dt_min < dt_max
        assert tol > 0.

        t = 0.
        dt.value = dt_min * dt_increase

        # Make sure initial time step does not exceed limits.
        dt.value = np.minimum(dt.value, dt_max)

        # Prepare output
        if self.output_xdmf is not None:
            self.output = np.atleast_1d(self.output_xdmf)

        it = 0

        ### DIRTY HACK HERE, sort in later!
        # ---------------------------------

        # Set up face measure at r=1
        fdim = self.mesh.topology.dim - 1
        facets = dfx.mesh.locate_entities(
            self.mesh, fdim, lambda x: np.isclose(x[0], 1.))
        facet_markers = np.full_like(facets, 1)

        facet_tag = dfx.mesh.meshtags(self.mesh, fdim, facets, facet_markers)

        dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

        A_p = sum(self.As)
        aas = self.As / A_p

        # The total particle surface across all cells
        A = self.comm.allreduce(A_p, op=MPI.SUM)

        # Ratio between cell surface and total surface
        a_p = A_p / A

        L_p = sum(aas * self.Ls)

        L = self.comm.allreduce(L_p * a_p)

        mus = self.u.sub(1).split()

        V_p_OCP_form_ufl = - 1 / L_p * sum(
            [L_ * a_ * mu_ * dA for L_, a_, mu_ in zip(self.Ls, aas, mus)])

        V_p_OCP_form = dfx.fem.form(V_p_OCP_form_ufl)

        while t < self.T_final:

            it += 1

            # if self.rt_analysis is not None:
            #     self.rt_analysis.analyze(t)

            self.u.x.scatter_forward()
            self.u0.x.array[:] = self.u.x.array[:]
            self.u0.x.scatter_forward()

            tol = 1e-6
            voltage_err = 1e99
            voltage_old = 0.
            it_voltage = 0
            max_it_voltage = 100

            iterations = 0  # counter for the inner iterations

            while (voltage_err > tol) and (it_voltage < max_it_voltage):

                it_voltage += 1

                # Compute cell and battery voltage
                # --------------------------------

                # cell open circuit voltage
                V_p_OCP = dfx.fem.assemble_scalar(V_p_OCP_form)

                # Global OCP across all cells
                V_OCP = self.comm.allreduce(a_p * L_p * V_p_OCP) / L

                # global voltage
                voltage = V_OCP - self.I_battery.value / L

                voltage_inc = abs(voltage - voltage_old)
                voltage_old = voltage

                # Update the cell current according to global voltage
                self.I_charge.value = L_p * (V_p_OCP - voltage)

                stop = self.experiment(t, cell_voltage=voltage)

                if stop:
                    break

                try:
                    if float(dt) < dt_min:

                        raise ValueError(f"Timestep too small (dt={dt.value})!")

                    try:
                        it_k, success = self.solver.solve(self.u)
                    except Exception as e:
                        log(e)
                        it_k = 9999

                    self.comm.barrier()
                    # log("Hit barrier")

                    iterations = max(it_k, iterations)

                    if not success:
                        raise RuntimeError("Newton solver did not converge.")
                    else:
                        iterations = MPI.COMM_WORLD.allreduce(
                            iterations, op=MPI.MAX)

                    # Adaptive timestepping a la Yibao Li et al. (2017)
                    # TODO: Timestepping through free energy
                    u_max_loc = np.abs(self.u.sub(0).x.array -
                                       self.u0.sub(0).x.array).max()

                    u_err_max = self.comm.allreduce(u_max_loc, op=MPI.MAX)

                    dt.value = min(max(tol / u_err_max, dt_min), dt_max, 1.1 * dt.value)

                    # callback(it, t, self.u)

                except StopEvent as e:

                    print(e)
                    print(">>> Stop integration.")

                    break

                except RuntimeError as e:

                    print(e)

                    # reset and continue with smaller time step.
                    self.u.x.array[:] = self.u0.x.array[:]

                    iterations = self.solver.max_iterations

                    if dt.value > dt_min:
                        dt.value *= 0.5

                        print(f"Decrease timestep to dt={dt.value:1.3e}")

                        continue
                    else:
                        if self.output is not None:
                            [o.save_snapshot(self.u, t) for o in self.output]

                except ValueError as e:

                    print(e)

                    if self.output is not None:
                        [o.save_snapshot(self.u, t, force=True)
                         for o in self.output]

                    break

            # Find the minimum timestep among all processes.
            # Note that we explicitly use COMM_WORLD since the mesh communicator
            # only groups the processes belonging to one particle.
            dt_global = MPI.COMM_WORLD.allreduce(dt.value, op=MPI.MIN)

            dt.value = dt_global

            t += float(dt)

            if self.output is not None:
                [o.save_snapshot(self.u, t) for o in self.output]

            if self.logging:
                perc = t / self.T_final * 100

                if MPI.COMM_WORLD.rank == 0:
                    print(
                        f"{perc:>3.0f} % :",
                        f"t[{it:06}] = {t:1.6f}, "
                        f"dt = {dt.value:1.3e}, "
                        f"its = {iterations}",
                        f"its_voltage = {it_voltage}",
                        f"voltage_inc = {voltage_inc}",
                        flush=True
                    )

        else:

            if self.output is not None:

                [o.finalize() for o in self.output]

        return


if __name__ == "__main__":

    comm_world = MPI.COMM_WORLD
    rank = comm_world.rank
    num_of_procs = comm_world.size

    # For the output file string
    digits = ceil(log10(num_of_procs))

    def info(*msg):
        return print(f"[{rank:>{digits}}]", *msg, flush=True)

    # %%
    # Discretization
    # --------------
    mesh_comm = MPI.COMM_SELF

    n_elem = 16
    mesh = dfx.mesh.create_unit_interval(mesh_comm, n_elem)
    exp_path = os.path.dirname(os.path.abspath(__file__))

    info(f"Initialize on {num_of_procs} processes.")

    Simulation.T_final = 0.1

    simulation = Simulation(
        mesh,
        num_particles=12,
        output_destination=exp_path +
        f"/simulation_output/output_{comm_world.rank:0{digits}}")

    simulation.run(tol=1e-5)
