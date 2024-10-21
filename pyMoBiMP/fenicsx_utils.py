import abc

import dolfinx as dfx

import h5py

import logging

from mpi4py import MPI

import numpy as np

import os

from petsc4py import PETSc

import shutil

from typing import Callable, List

import ufl


logger = logging.getLogger(__name__)


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


def get_mesh_spacing(mesh, return_full=False):

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim, np.array(range(num_cells)))

    dx_cell = h.min()

    if return_full:
        return h
    else:
        return dx_cell


def time_stepping(
    solver,
    u,
    u0,
    T,
    dt,
    t_start=0,
    dt_max=10.0,
    dt_min=1e-9,
    dt_increase=1.1,
    tol=1e-6,
    event_handler=lambda t, **pars: None,
    output=None,
    runtime_analysis=None,
    logging=True,
    callback=lambda it, t, u: None,
    **event_pars,
):

    assert dt_min < dt_max
    assert tol > 0.
    assert dt_increase > 1.

    t = t_start
    dt.value = dt_min * dt_increase

    # Make sure initial time step does not exceed limits.
    dt.value = np.minimum(dt.value, dt_max)

    # Prepare outout
    if output is not None:
        output = np.atleast_1d(output)

    it = 0

    while t < T:

        it += 1

        if runtime_analysis is not None:
            runtime_analysis.analyze(t)

        try:
            u.x.scatter_forward()
            u0.x.array[:] = u.x.array[:]
            u0.x.scatter_forward()

            if runtime_analysis is not None:
                voltage = runtime_analysis.data[-1][-1]
            else:
                voltage = 0.

            stop = event_handler(t, cell_voltage=voltage, **event_pars)

            if stop:
                break

            if float(dt) < dt_min:

                raise ValueError(f"Timestep too small (dt={dt.value})!")

            iterations, success = solver.solve(u)

            if not success:
                raise RuntimeError("Newton solver did not converge.")

            # Adaptive timestepping a la Yibao Li et al. (2017)
            u_max_loc = np.abs(u.sub(0).x.array - u0.sub(0).x.array).max()

            u_err_max = u.function_space.mesh.comm.allreduce(u_max_loc, op=MPI.MAX)

            if iterations < solver.max_it / 5:
                # Use the given increment factor if we are in a safe region, i.e.,
                # if the Newton solver converges sufficiently fast.
                inc_factor = dt_increase
            elif iterations < solver.max_it / 2:
                # Reduce the increment if we take more iterations.
                inc_factor = 1 + 0.1 * (dt_increase - 1)
            elif iterations > solver.max_it * 0.8:
                # Reduce the timestep in case we are approaching max_it
                inc_factor = 0.9
            else:
                # Do not increase timestep between [0.5*max_it, 0.8*max_it]
                inc_factor = 1.0

            dt.value = min(
                           max(tol / u_err_max, dt_min),
                           dt_max,
                           inc_factor * dt.value)

            callback(it, t, u)

        except StopEvent as e:

            print(e)
            print(">>> Stop integration.")

            break

        except RuntimeError as e:

            print(e)

            # reset and continue with smaller time step.
            u.x.array[:] = u0.x.array[:]

            iterations = solver.max_it

            if dt.value > dt_min:
                dt.value *= 0.5

                print(f"Decrease timestep to dt={dt.value:1.3e}")

                continue

            else:
                if output is not None:

                    [o.save_snapshot(u, t) for o in output]

        except ValueError as e:

            print(e)

            if output is not None:
                [o.save_snapshot(u, t, force=True) for o in output]

            break

        # Find the minimum timestep among all processes.
        # Note that we explicitly use COMM_WORLD since the mesh communicator
        # only groups the processes belonging to one particle.
        dt_global = MPI.COMM_WORLD.allreduce(dt.value, op=MPI.MIN)

        dt.value = dt_global

        t += float(dt)

        if output is not None:
            [o.save_snapshot(u, t) for o in output]

        if logging:
            perc = (t - t_start) / (T - t_start) * 100

            if MPI.COMM_WORLD.rank == 0:
                print(
                    f"{perc:>3.0f} % :",
                    f"t[{it:06}] = {t:1.6f}, "
                    f"dt = {dt.value:1.3e}, "
                    f"its = {iterations}",
                    flush=True
                )

    else:

        if output is not None:

            [o.finalize() for o in output]

    return


class NonlinearProblem:
    """
    Custom implementation of NonlinearProblem to make sure we have a Jacobian.
    """

    def __init__(
        self, F: ufl.Form, c: dfx.fem.Function, bcs: List[dfx.fem.DirichletBC] = []
    ):

        V = c.function_space
        self.mesh_comm = V.mesh.comm

        dc = ufl.TrialFunction(V)

        J = ufl.derivative(F, c, dc)

        self.L = dfx.fem.form(F)
        self.a = dfx.fem.form(J)

        self.bcs = bcs

    def pack_constants_and_coeffs(self):

        constants_L = [
            form and dfx.cpp.fem.pack_constants(form._cpp_object) for form in [self.L]
        ]
        coeffs_L = [
            dfx.cpp.fem.pack_coefficients(form._cpp_object) for form in [self.L]]

        constants_a = [
            [
                dfx.cpp.fem.pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in [[self.a]]
        ]

        coeffs_a = [
            [
                {} if form is None else dfx.cpp.fem.pack_coefficients(form._cpp_object)
                for form in forms
            ]
            for forms in [[self.a]]
        ]

        return dict(coeffs_a=coeffs_a,
                    constants_a=constants_a,
                    coeffs_L=coeffs_L,
                    constants_L=constants_L)

    def scatter_Function_to_vector(self, w: dfx.fem.Function, x: PETSc.Vec):
        # Scatter previous solution `w` to `self.x`, the blocked version used for lifting

        dfx.cpp.la.petsc.scatter_local_vectors(
            x,
            [si.x.petsc_vec.array_r for si in [w]],
            [
                (
                    si.function_space.dofmap.index_map,
                    si.function_space.dofmap.index_map_bs,
                )
                for si in [w]
            ],
        )
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def update_solution(self, dx: PETSc.Vec, beta: float, w: dfx.fem.Function):
        # Update solution
        offset_start = 0
        for s in [w]:
            num_sub_dofs = (
                s.function_space.dofmap.index_map.size_local
                * s.function_space.dofmap.index_map_bs
            )

            s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                beta * dx.array_r[offset_start:offset_start + num_sub_dofs]
            )
            s.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            offset_start += num_sub_dofs

    def form(self, x: PETSc.Vec):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble residual vector."""

        with b.localForm() as b_local:
            b_local.set(0.0)

        dfx.fem.petsc.assemble_vector(b, self.L)

        dfx.fem.petsc.apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)

        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        dfx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble Jacobian matrix."""

        A.zeroEntries()
        dfx.fem.petsc.assemble_matrix(A, self.a, bcs=self.bcs)
        A.assemble()

    def matrix(self):
        return dfx.fem.petsc.create_matrix(self.a)

    def vector(self):
        return dfx.fem.petsc.create_vector(self.L)


class NonlinearProblemBlock:
    """
    Block-based implementation of NonlinearProblem to make sure we have a Jacobian.
    """

    def __init__(
        self,
        F: list[ufl.Form],
        w: list[dfx.fem.Function],
        bcs: list[dfx.fem.DirichletBC] = [],
    ):

        dw = [ufl.TrialFunction(c.function_space) for c in w]

        J = [[ufl.derivative(Fi, c, dc) for c, dc in zip(w, dw)] for Fi in F]

        self.L = dfx.fem.form(F)
        self.a = dfx.fem.form(J)

        self.bcs = bcs

    def pack_constants_and_coeffs(self):
        # Pack constants and coefficients
        constants_L = [
            form and dfx.cpp.fem.pack_constants(form._cpp_object) for form in self.L
        ]
        coeffs_L = [
            dfx.cpp.fem.pack_coefficients(form._cpp_object) for form in self.L]

        constants_a = [
            [
                dfx.cpp.fem.pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in self.a
        ]

        coeffs_a = [
            [
                {} if form is None else dfx.cpp.fem.pack_coefficients(form._cpp_object)
                for form in forms
            ]
            for forms in self.a
        ]

        return dict(coeffs_a=coeffs_a,
                    constants_a=constants_a,
                    coeffs_L=coeffs_L,
                    constants_L=constants_L)

    def scatter_Function_to_vector(self, w: List[dfx.fem.Function], x: PETSc.Vec):
        # Scatter previous solution `w` to `self.x`, the blocked version used for lifting
        dfx.cpp.la.petsc.scatter_local_vectors(
            x,
            [si.x.petsc_vec.array_r for si in w],
            [
                (
                    si.function_space.dofmap.index_map,
                    si.function_space.dofmap.index_map_bs,
                )
                for si in w
            ],
        )
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def update_solution(self,
                        dx: PETSc.Vec,
                        beta: float,
                        w: List[dfx.fem.Function]):

        offset_start = 0
        for s in w:
            num_sub_dofs = (
                s.function_space.dofmap.index_map.size_local
                * s.function_space.dofmap.index_map_bs
            )

            s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                beta * dx.array_r[offset_start:offset_start + num_sub_dofs]
            )
            s.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            offset_start += num_sub_dofs

    def form(self, x: PETSc.Vec):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble residual vector."""

        with b.localForm() as b_local:
            b_local.set(0.0)

        dfx.fem.petsc.assemble_vector_block(
            b,
            self.L,
            self.a,
            self.bcs,
            x0=x,
            scale=-1,
            **self.pack_constants_and_coeffs()
            )

        b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble Jacobian matrix."""

        A.zeroEntries()
        dfx.fem.petsc.assemble_matrix_block(A, self.a, bcs=self.bcs)
        A.assemble()

    def matrix(self):
        return dfx.fem.petsc.create_matrix_block(self.a)

    def vector(self):
        return dfx.fem.petsc.create_vector_block(self.L)


class NewtonSolver():
    """
    Custom block-based Newton solver inspired by scifem's NewtonSolver
    implementation.
    """

    def __init__(self,
                 comm: MPI.Intracomm,
                 problem: NonlinearProblemBlock,
                 max_iterations: int = 10,
                 rtol: float = 1e-10,
                 beta: float = 1.0,
                 error_on_nonconvergence: bool = True):

        self.comm = comm  # NOTE: This communicator is not used by this class.

        self.problem = problem

        self.A = problem.matrix()
        self.L = problem.vector()
        self.x = problem.vector()
        self.dx = problem.vector()

        # Store accessible/modifiable properties
        self._pre_solve_callback = None
        self._post_solve_callback = None
        self.beta = beta
        self._error_on_nonconvergence = error_on_nonconvergence

        self.max_it = max_iterations
        self.rtol = rtol
        self.convergence_criterion = "incremental"

        self.ksp = PETSc.KSP()
        self.krylov_solver = self.ksp.create(self.L.getComm().tompi4py())
        self.krylov_solver.setOperators(self.A)

        self.krylov_solver_setup()

    def krylov_solver_setup(self):
        # Set default options for the linear solver
        ksp = self.ksp
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.getPC().setFactorSetUpSolverType()
        ksp.getPC().setFactorSetUpSolverType()

        self.krylov_solver.setFromOptions()
        self.A.setFromOptions()
        self.L.setFromOptions()

    def set_pre_solve_callback(self, callback: Callable[["NewtonSolver"], None]):
        """Set a callback function that is called before each Newton iteration."""
        self._pre_solve_callback = callback

    def set_post_solve_callback(self, callback: Callable[["NewtonSolver"], None]):
        """Set a callback function that is called after each Newton iteration."""
        self._post_solve_callback = callback

    def set_x(self, w: dfx.fem.Function | List[dfx.fem.Function]):
        self.problem.scatter_Function_to_vector(w, self.x)

    def setF(self, x: PETSc.Vec):
        # Assemble the residual vector
        self.problem.F(x, self.L)

    def setJ(self, x: PETSc.Vec):
        self.problem.J(x, self.A)

    def set_form(self, x: PETSc.Vec):
        self.problem.form(x)

    def update_solution(self, w: dfx.fem.Function | List[dfx.fem.Function]):
        self.problem.update_solution(self.dx, self.beta, w)

    def solve(self, w: dfx.fem.Function | List[dfx.fem.Function]):

        for it in range(self.max_it):

            if self._pre_solve_callback is not None:
                self._pre_solve_callback(self)

            self.set_x(w)

            # Assemble RHS
            self.setF(self.x)

            # Assemble the Jacobian
            self.setJ(self.x)

            # Finally update ghost values
            self.set_form(self.x)

            # Solve linear problem
            self.krylov_solver.solve(self.L, self.dx)

            if self._error_on_nonconvergence:
                if (status := self.krylov_solver.getConvergedReason()) <= 0:
                    raise RuntimeError(
                        f"Linear solver did not converge, got reason: {status}")

            # Compute norm of update
            correction_norm = self.dx.norm(0)

            if np.isnan(correction_norm) or np.isinf(correction_norm):
                raise RuntimeError("NaNs in NewtonSolver!")

            self.update_solution(w)
            it += 1

            if self._post_solve_callback is not None:
                self._post_solve_callback(self)

            # print(f"Iteration {it}: Correction norm {correction_norm}")
            if self.convergence_criterion == 'incremental':

                logger.info(f"Iteration {it}: |dx| = {correction_norm:1.3e}")

                if correction_norm < self.rtol * self.x.norm(0):
                    return it, True

            elif self.convergence_criterion == "residual":
                if self.L.norm(0) < self.rtol:
                    return it, True

            elif self.convergence_criterion == "none":
                if it == self.max_it:
                    return it, True

            else:
                raise ValueError(
                    f"Convergence criterion `{self.convergence_criterion}` not suported")

        if self._error_on_convergence:
            raise RuntimeError("Newton solver did not converge")
        else:
            return it, False


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

    def save_snapshot(self, u_state, t, force=False):

        if t >= self.t_out_next or force:

            print(f">>> Save snapshot [{self.it_out:04}] t={t:1.3f}")

            self.output_container.append(self.extract_output(u_state, t))
            self.output_times.append(t)

            self.t_out_last = self.t_out_next

            # Increase the next time step except for the last one.
            if self.it_out < len(self.ts_out_planned):
                self.t_out_next = self.ts_out_planned[self.it_out]
            self.it_out += 1

    @abc.abstractmethod
    def extract_output(self, u_state, t):
        pass

    def get_output(self, return_time=False):

        if return_time:
            return self.output_times, self.output_container
        else:
            return self.output_container


class VTXOutput(OutputBase):

    def setup(self, filename="output.bp", variable_transform=lambda y: y):

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

    def save_snapshot(self, u_state, t, force=False):
        super().save_snapshot(u_state, t, force)

        if t >= self.t_out_next or force:
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
        V = self.u_state.function_space

        num_vars = V.num_sub_spaces

        # At top level, we should have two spaces, one for c and one for mu.
        assert num_vars == 2

        ret = []

        for i in range(num_vars):

            V_sub, _ = V.sub(i).collapse()

            if i == 0:
                name = "y"
            elif i == 1:
                name = "mu"
            else:
                raise ValueError(f"No component with index {i} available!")

            num_comp = V_sub.num_sub_spaces

            # If we have a single-particle space, we are already at the lowest
            # level where the individual functions for c and mu live.

            if num_comp == 0:
                func = self.u_state.sub(i)
                func.name = name

                ret.append(func)

            else:
                for j in range(num_comp):

                    func = self.u_state.sub(i).sub(j)
                    func.name = name + f"_{j}"

                    ret.append(func)

        return ret

    def save_snapshot(self, u_state, t, force=False):

        if t >= self.t_out_next or force:

            print(f">>> Save snapshot [{self.it_out:04}] t={t:1.3f}")

            self.t_out_last = self.t_out_next

            # Increase the next time step except for the last one.
            if self.it_out < len(self.ts_out_planned):
                self.t_out_next = self.ts_out_planned[self.it_out]
            self.it_out += 1

            output = self.extract_output(self.u_state, t)
            with self.FileType(self.comm, self.filename, "a") as file:
                [file.write_function(comp, t) for comp in output]

    def get_output(self, return_time=False):
        raise NotImplementedError(
            f"In {self.__class__}, get_output is not implemented!")

    def finalize(self):
        # File status should be clear since we use always use a context manager.
        pass


class Fenicx1DOutput(OutputBase):

    def setup(self, x):

        V = self.u_state.function_space

        mesh = V.mesh

        points_on_proc, cells = evaluation_points_and_cells(mesh, x)

        self.x_eval, self.cells = points_on_proc, cells

    def finalize(self):
        return super().finalize()

    def extract_output(self, u_state, t):

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

    def __init__(self, u_state, *args, filename=None, **kwargs):

        # Attach the Function object
        self.u_state = u_state

        # Initialize empty containers
        self.t = []
        self.data = []

        self.filename = filename

        self.setup(u_state, *args, **kwargs)

        # Touch the file to make sure it exists.
        if self.filename is not None:
            with open(self.filename, "w"):
                pass

    @abc.abstractmethod
    def setup(self, u_state, *args, **kwargs):
        pass

    @abc.abstractmethod
    def analyze(self, t):

        self.t.append(t)

        if self.filename is not None and MPI.COMM_WORLD.rank == 0:
            with open(self.filename, "a") as file:
                np.savetxt(file, np.array([[t, *self.data[-1]]]))


class StopEvent(Exception):
    pass


def strip_off_xdmf_file_ending(file_name, ending=""):

    # Strip off the file ending for uniform file handling
    if file_name[-3:] == ".h5":
        file_name_base = file_name[:-3]

    elif file_name[-5:] == ".xdmf":
        file_name_base = file_name[:-5]

    elif file_name[-len(ending):] == ending:
        file_name_base = file_name[:-len(ending)]

    else:
        if "." in os.path.basename(file_name):
            raise ValueError(f"Unrecognized file ending: {file_name}!")
        else:
            file_name_base = file_name

    return file_name_base


class SimulationFile(h5py.File):
    """A file handler class to open pyMoBiMP simulation output.

    The main purpose of the class is to wrap a copy operation to the
    standard h5py file handler to avoid deadlocks when opening files
    from a running simulation.

    Attributes
    ----------
    _file_name : str
        file name pointing to the simulation output
    _file_name_tmp : str
        the temporary file name of the copied file
    """

    def __init__(self, file_name):
        """Construct file name and temporary file name.

        Parameters
        ----------
        file_name : str | pathlib.Path
            File name pointing to the file name base or XDMF or H5 file, the
            ending can be omitted.
        """
        file_name_base = strip_off_xdmf_file_ending(file_name)

        self._file_name = file_name_base + ".h5"
        self._file_name_tmp = file_name_base + "_tmp" + ".h5"

        # To avoid file locks, copy the current version of the file to a tmp file.
        shutil.copy(self._file_name, self._file_name_tmp)

        # Advise base class to open the tmp file.
        super().__init__(self._file_name_tmp)

    def __exit__(self, *args, **kwargs):
        """
        Ensures that after the file operation is done, the temporary file is deleted.
        """

        super().__exit__(self, *args, **kwargs)

        # ... and remove it
        os.remove(self._file_name_tmp)


def get_particle_number_from_mesh(mesh):
    from pyMoBiMP.cahn_hilliard_utils import (
        create_particle_summation_measure)

    dA = create_particle_summation_measure(mesh)

    nop_form = dfx.fem.form(dfx.fem.Constant(mesh, 1.0) * dA)
    num_particles_from_mesh = int(round(dfx.fem.assemble_scalar(nop_form)))

    return num_particles_from_mesh


def read_data(filebasename: str,
              comm: MPI.Intracomm = MPI.COMM_WORLD,
              return_grid: bool = False):

    mesh_file = strip_off_xdmf_file_ending(filebasename) + ".xdmf"

    with dfx.io.XDMFFile(comm, mesh_file, 'r') as file:
        try:
            mesh = file.read_mesh()
        except ValueError:
            mesh = file.read_mesh(name="Grid")

    print(f"Read data from {filebasename} ...")

    with SimulationFile(filebasename) as f:
        print(f["Function"].keys())

        num_particles = len(f["Function"].keys()) // 2

        # grid coordinates
        if "mesh" in f["Mesh"].keys():
            x_data = f["Mesh/mesh/geometry"][()]
        elif "Grid" in f["Mesh"].keys():
            x_data = f["Mesh/Grid/geometry"][()]
        else:
            raise ValueError("Neither 'Mesh' nor 'Grid' detected in " + filebasename)

        if "y_0" in f["Function"].keys():
            t_keys = f["Function/y_0"].keys()
        elif "y" in f["Function"].keys():
            t_keys = f["Function/y"].keys()
        else:
            raise KeyError("No appropriate key found in 'f'!")

        # time steps (convert from string to float)
        t = [float(t.replace("_", ".")) for t in t_keys]

        # list of data stored as numpy arrays
        if "y_0" in f["Function"].keys():
            u_data = np.array(
                [
                    [
                        (
                            f[f"Function/y_{i_part}"][u_key][()].squeeze(),
                            f[f"Function/mu_{i_part}"][u_key][()].squeeze(),
                        )
                        for i_part in range(num_particles)
                    ]
                    for u_key in t_keys
                ]
            )
        else:
            u_data = np.array(
                [
                    [
                        f["Function/y"][u_key][()].squeeze(),
                        f["Function/mu"][u_key][()].squeeze(),
                    ]
                    for u_key in t_keys
                ]
            )

    # Catch the DFN mesh case
    if num_particles == 1:
        num_particles = get_particle_number_from_mesh(mesh)

        u_data = u_data.reshape(len(t), 2, num_particles, -1)
        u_data = u_data.transpose(0, 2, 1, 3)

        x_data = x_data.reshape(num_particles, -1, 2).transpose((-1, 0, 1))

        sorted_indx = np.argsort(x_data[1, :, 0])

        x_data = x_data[:, sorted_indx, :]
        u_data = u_data[:, sorted_indx, :, :]

    else:
        x_data = x_data.T

    print(f"Found {num_particles} particles.")

    # It is necessary to sort the input by the time.
    sorted_indx = np.argsort(t)

    t = np.array(t)[sorted_indx]
    u_data = np.array(u_data)[sorted_indx]

    filebasename = strip_off_xdmf_file_ending(filebasename)

    # Read the runtime analysis output.
    rt_data = np.loadtxt(filebasename + "_rt.txt")

    # Check the output shape of the array:
    # Dimensions are [time, num_particles, variable_name, radius]
    # assert u_data.shape[:3] == (len(t), num_particles, 2)

    if return_grid:
        return (num_particles, t, x_data, u_data, rt_data), mesh

    else:
        return num_particles, t, x_data, u_data, rt_data
