import abc

import dolfinx as dfx
from dolfinx.nls.petsc import NewtonSolver as NewtonSolverBase

import numpy as np

from petsc4py import PETSc


def evaluation_points_and_cells(mesh, x):
    """points_on_proc and cells to be used with dfx.fem.Function.eval()."""

    points = np.zeros((3, len(x)))
    points[0] = x

    bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dfx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points.T
    )
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    return points_on_proc, cells


def get_mesh_spacing(mesh):

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim, range(num_cells))

    dx_cell = h.min()

    return dx_cell


def time_stepping(
    solver,
    u,
    u0,
    T,
    dt,
    t_start=0,
    dt_max=10.0,
    dt_min=1e-7,
    dt_increase=1.01,
    event_handler=lambda t, u, **pars: None,
    output=None,
    runtime_analysis=None,
    logging=True,
    **event_pars,
):

    t = t_start

    # Make sure initial time step does not exceed limits.
    dt.value = np.minimum(dt.value, dt_max)

    # Prepare outout
    if output is not None:
        output = np.atleast_1d(output)

    while t < T:

        try:
            u0.x.array[:] = u.x.array[:]

            stop = event_handler(t, u, **event_pars)

            if stop:
                break

            if float(dt) < dt_min:

                raise ValueError(f"Timestep too small (dt={dt.value})!")

            iterations, success = solver.solve(u)

        except StopEvent as e:

            # TODO: Fix the event handler.

            print(e)
            print(">>> Stop integration.")

            break

        except RuntimeError as e:

            print(e)

            # reset and continue with smaller time step.
            u.x.array[:] = u0.x.array[:]

            if dt.value > dt_min:
                dt.value *= 0.5

                print(f"Decrease timestep to dt={dt.value:1.3e}")

                continue

            else:
                if output is not None:

                    output.save_snapshot(u, t)

        except ValueError as e:

            print(e)

            break

        if output is not None:
            [o.save_snapshot(u, t) for o in output]

        if runtime_analysis is not None:
            runtime_analysis.analyze(u, t)

        t += float(dt)

        if dt.value * dt_increase < dt_max:
            dt.value *= dt_increase

        if logging:
            print(f"t = {t:1.6f} : dt = {dt.value:1.3e}, its = {iterations}")

    else:

        if output is not None:

            [o.finalize() for o in output]

    return


class NewtonSolver(NewtonSolverBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.convergence_criterion = "incremental"
        self.rtol = 1e-9

        # # We can customize the linear solver used inside the NewtonSolver by
        # # modifying the PETSc options
        ksp = self.krylov_solver
        opts = PETSc.Options()  # type: ignore
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        ksp.setFromOptions()


class OutputBase(abc.ABC):

    def __init__(self, u_state, ts_out, *args, **kwargs):

        # Store the state and the output times.
        self.u_state = u_state
        self.ts_out_planned = ts_out

        # output counter
        self.it_out = 0
        self.t_out_last = None
        self.t_out_next = ts_out[0]

        self.output_container = []
        self.output_times = []

        self.setup(*args, **kwargs)

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass

    def save_snapshot(self, u_state, t):

        if t >= self.t_out_next:

            print(f">>> Save snapshot [{self.it_out:04}] t={t:1.3f}")

            self.output_container.append(self.extract_output(u_state, t))
            self.output_times.append(t)

            self.it_out += 1
            self.t_out_last = self.t_out_next
            self.t_out_next = self.ts_out_planned[self.it_out]

    @abc.abstractmethod
    def extract_output(self, u_state, t):
        pass

    def get_output(self, return_time=False):

        # TODO: add mesh mesh information.

        if return_time:
            return self.output_times, self.output_container
        else:
            return self.output_container


class VTXOutput(OutputBase):

    def setup(self, filename="output.bp", variable_transform = lambda y: y):

        mesh = self.u_state.function_space.mesh
        comm = mesh.comm

        self.c_of_y = variable_transform

        output = self.extract_output(self.u_state, 0.)

        self.writer = dfx.io.VTXWriter(comm, filename, output)

    def extract_output(self, u_state, t):

        V = self.u_state.function_space

        num_comp = V.num_sub_spaces

        ret = []

        for i in range(num_comp):

            i_comp = self.u_state.sub(i)
            i_comp.name = f"comp_{i}"

            if i == 0:

                V0, _ = V.sub(i).collapse()

                i_comp_transformed = dfx.fem.Function(V0)
                i_comp_transformed.name = i_comp.name

                i_comp_transformed.interpolate(
                    dfx.fem.Expression(self.c_of_y(i_comp),
                                       V0.element.interpolation_points())
                )

                # Overwrite the reference that is been put out
                i_comp = i_comp_transformed

            ret.append(i_comp)

        return ret

    def save_snapshot(self, u_state, t):
        super().save_snapshot(u_state, t)

        if t >= self.t_out_next:
            self.writer.write(t)

    def get_output(self, return_time=False):
        raise NotImplementedError("In VTXOutput, get_output is not implemented!")

    def finalize(self):

        self.writer.close()


class FileOutput(OutputBase):

    FileType = dfx.io.XDMFFile

    def setup(self, filename="output", variable_transform=lambda y: y):

        mesh = self.u_state.function_space.mesh
        self.comm = comm = mesh.comm

        self.c_of_y = variable_transform

        self.filename = filename

        with self.FileType(comm, self.filename, "w") as file:
            file.write_mesh(mesh)

    def extract_output(self, u_state, t):
        # Borrow from VTXOutput.
        return VTXOutput.extract_output(self, u_state, t)

    def save_snapshot(self, u_state, t):

        if t >= self.t_out_next:

            print(f">>> Save snapshot [{self.it_out:04}] t={t:1.3f}")

            self.it_out += 1
            self.t_out_last = self.t_out_next
            self.t_out_next = self.ts_out_planned[self.it_out]

            output = self.extract_output(self.u_state, t)
            with self.FileType(self.comm, self.filename, "a") as file:
                [file.write_function(comp, t) for comp in output]

    def get_output(self, return_time=False):
        raise NotImplementedError(
            f"In {self.__class__}, get_output is not implemented!")

    def finalize(self):
        pass  # File status should be clear since we
              # use always use a context manager.


class Fenicx1DOutput(OutputBase):

    def setup(self, x):

        V = self.u_state.function_space

        mesh = V.mesh

        points_on_proc, cells = evaluation_points_and_cells(mesh, x)

        self.x_eval, self.cells = points_on_proc, cells

    def finalize(self):
        return super().finalize()

    def extract_output(self, u_state, t):

        # TODO: Make sure self.u_state and u_state are consistent.
        V = self.u_state.function_space

        num_comp = V.num_sub_spaces

        output_snapshot = []

        for i_comp in range(num_comp):

            values = u_state.sub(i_comp).eval(self.x_eval, self.cells)

            output_snapshot.append(values)

        return output_snapshot

    def get_output(self, return_time=False, return_coords=False):

        ret = list(super().get_output(return_time))

        if return_coords:
            ret = [self.x_eval, *ret]

        return ret


class RuntimeAnalysisBase(abc.ABC):

    def __init__(self, *args, **kwargs):

        # Initialize empty containers
        self.t = []
        self.data = []

        self.setup(*args, **kwargs)

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def analyze(self, u_state, t):

        self.t.append(t)


class StopEvent(Exception):
    pass
