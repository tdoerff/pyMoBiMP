# %%
import dolfinx as dfx

from mpi4py import MPI

from math import log10, ceil

import numpy as np

import os

from pyMoBiMP.cahn_hilliard_utils import (
    MultiParticleSimulation as SimulationBase,
)

from pyMoBiMP.fenicsx_utils import StopEvent


class Simulation(SimulationBase):

    def __init__(self, *args, logging=True, **kwargs):

        self.logging = logging

        super().__init__(*args, **kwargs)

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

        while t < self.T_final:

            it += 1

            if self.rt_analysis is not None:
                self.rt_analysis.analyze(t)

            try:
                self.u.x.scatter_forward()
                self.u0.x.array[:] = self.u.x.array[:]
                self.u0.x.scatter_forward()

                if self.rt_analysis is not None:
                    voltage = self.rt_analysis.data[-1][-1]
                else:
                    voltage = 0.

                stop = self.experiment(t, cell_voltage=voltage)

                if stop:
                    break

                if float(dt) < dt_min:

                    raise ValueError(f"Timestep too small (dt={dt.value})!")

                iterations, success = self.solver.solve(self.u)

                if not success:
                    raise RuntimeError("Newton solver did not converge.")
                else:
                    iterations = MPI.COMM_WORLD.allreduce(iterations, op=MPI.MAX)

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
                    [o.save_snapshot(self.u, t, force=True) for o in self.output]

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
