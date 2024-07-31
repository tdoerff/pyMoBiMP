import abc

import dolfinx as dfx

import h5py

from mpi4py import MPI

import numpy as np

import os

from petsc4py import PETSc

import shutil

from typing import List

import ufl


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
    h = mesh.h(tdim, range(num_cells))

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
    event_handler=lambda t, u, **pars: None,
    output=None,
    runtime_analysis=None,
    logging=True,
    callback=lambda it, t, u: None,
    **event_pars,
):

    assert dt_min < dt_max
    assert tol > 0.

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
            else:
                iterations = MPI.COMM_WORLD.allreduce(iterations, op=MPI.MAX)

            # Adaptive timestepping a la Yibao Li et al. (2017)
            # TODO: Timestepping through free energy
            u_max_loc = np.abs(u.sub(0).x.array - u0.sub(0).x.array).max()

            u_err_max = u.function_space.mesh.comm.allreduce(u_max_loc, op=MPI.MAX)

            dt.value = min(max(tol / u_err_max, dt_min), dt_max, 1.1 * dt.value)

            callback(it, t, u)

        except StopEvent as e:

            print(e)
            print(">>> Stop integration.")

            break

        except RuntimeError as e:

            print(e)

            # reset and continue with smaller time step.
            u.x.array[:] = u0.x.array[:]

            iterations = solver.max_iterations

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

    def __init__(self, F, c, bcs=[]):

        V = c.function_space
        self.mesh_comm = V.mesh.comm

        dc = ufl.TrialFunction(V)

        J = ufl.derivative(F, c, dc)

        self.L = dfx.fem.form(F)
        self.a = dfx.fem.form(J)

        self.bcs = bcs

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """Assemble residual vector."""

        with b.localForm() as b_local:
            b_local.set(0.0)

        dfx.fem.petsc.assemble_vector(b, self.L)

        dfx.fem.petsc.apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)

        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        dfx.fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, x, A):
        """Assemble Jacobian matrix."""

        A.zeroEntries()
        dfx.fem.petsc.assemble_matrix(A, self.a, bcs=self.bcs)
        A.assemble()

    def matrix(self):
        return dfx.fem.petsc.create_matrix(self.a)

    def vector(self):
        return dfx.fem.petsc.create_vector(self.L)


class BlockNonlinearProblem:
    """
    Wrapper class to collect independent block problems to be solved simultaneously.
    """
    def __init__(self,
                 Fs: List[ufl.Form],
                 cs: List[dfx.fem.Function],
                 bcss: List[List[dfx.fem.DirichletBC | None]] = []):

        assert len(Fs) == len(cs)

        self.num_of_blocks = len(Fs)

        if len(bcss) == 0:
            bcss = [[], ] * self.num_of_blocks

        assert len(bcss) == self.num_of_blocks

        self.problems = [
            NonlinearProblem(F, c, bcs) for F, c, bcs in zip(Fs, cs, bcss)]


class NewtonSolver():

    def __init__(self,
                 comm,
                 problem,
                 max_iterations=10,
                 rtol=1e-10,
                 callback=lambda solver, uh: None):

        self.comm = comm

        self.problem = problem

        self.A = problem.matrix()
        self.L = problem.vector()

        self.max_it = max_iterations
        self.rtol = rtol
        self.convergence_criterion = "incremental"

        self.callback = callback

        self.ksp = PETSc.KSP()
        self.krylov_solver = self.ksp.create(comm)
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

    def setF(self, x):
        # Assemble the residual vector
        self.problem.F(x, self.L)

        # Scale residual by -1
        self.L.scale(-1)
        self.L.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD)

    def setJ(self, x):
        self.problem.J(x, self.A)

    def set_form(self, x):
        self.problem.form(x)

    def solve(self, ch):

        V = ch.function_space

        dc = dfx.fem.Function(V)

        success = False

        for it in range(self.max_it):

            self.callback(self, ch)

            # Assemble RHS
            self.setF(dc.vector)

            # Assemble the Jacobian
            self.setJ(dc.vector)

            # Finally update ghost values
            self.set_form(dc.vector)

            # Solve linear problem
            self.krylov_solver.solve(self.L, dc.vector)

            # Compute norm of update
            correction_norm = dc.vector.norm(0)

            if np.isnan(correction_norm) or np.isinf(correction_norm):
                raise RuntimeError("NaNs in NewtonSolver!")

            dc.x.scatter_forward()
            # Update u_{i+1} = u_i + delta u_i
            ch.x.array[:] += dc.x.array
            it += 1

            # print(f"Iteration {it}: Correction norm {correction_norm}")
            if self.convergence_criterion == 'incremental':
                if correction_norm < self.rtol * ch.vector.norm(0):
                    success = True
                    break

            elif self.convergence_criterion == "residual":
                if self.L.norm(0) < self.rtol:
                    success = True
                    break

            elif self.convergence_criterion == "none":
                if it == self.max_it:
                    success = True
                    break

            else:
                raise ValueError(
                    f"Convergence criterion `{self.convergence_criterion}` not suported")

        return it, success


class BlockNewtonSolver:
    def __init__(self,
                 comm,
                 block_problem,
                 max_iterations=50,
                 rtol=1e-10,
                 convergence_criterion="incremental",
                 callback=lambda solver, uh: None):

        self.comm = comm

        self.problem = block_problem

        self.max_it = max_iterations
        self.rtol = rtol
        self.convergence_criterion = convergence_criterion
        self.callback = callback

        self.block_solvers = [
            NewtonSolver(comm, problem) for problem in self.problem.problems]

    def solve(self, chs):

        Vs = [ch.function_space for ch in chs]

        dcs = [dfx.fem.Function(V) for V in Vs]

        success = False

        for it in range(self.max_it):

            correction_norm = 0.
            solution_norm = 0.
            residual_norm = 0.

            self.callback(self, chs)

            for ch, dc, solver in zip(chs, dcs, self.block_solvers):

                # Assemble RHS
                solver.setF(dc.vector)

                # Assemble the Jacobian
                solver.setJ(dc.vector)

                # Finally update ghost values
                solver.set_form(dc.vector)

                # Solve linear problem
                solver.krylov_solver.solve(solver.L, dc.vector)

                if np.isnan(correction_norm) or np.isinf(correction_norm):
                    raise RuntimeError("NaNs in NewtonSolver!")

                dc.x.scatter_forward()
                # Update u_{i+1} = u_i + delta u_i
                ch.x.array[:] += dc.x.array
                it += 1

                # Compute norm of update
                correction_norm += dc.vector.norm(0)
                solution_norm += ch.vector.norm(0)
                residual_norm += solver.L.norm(0)

                print(it, correction_norm, residual_norm)

            # print(f"Iteration {it}: Correction norm {correction_norm}")
            if self.convergence_criterion == 'incremental':
                if correction_norm < self.rtol * solution_norm:
                    success = True
                    break

            elif self.convergence_criterion == "residual":
                if residual_norm < self.rtol:
                    success = True
                    break

            elif self.convergence_criterion == "none":
                if it == self.max_it:
                    success = True
                    break

            else:
                raise ValueError(
                    f"Convergence criterion `{self.convergence_criterion}` not suported")

        return it, success


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

    def __init__(self, u_state, *args, **kwargs):

        # Attach the Function object
        self.u_state = u_state

        # Initialize empty containers
        self.t = []
        self.data = []

        self.filename = None

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


def strip_off_xdmf_file_ending(file_name):

    # Strip off the file ending for uniform file handling
    if file_name[-3:] == ".h5":
        file_name_base = file_name[:-3]

    elif file_name[-5:] == ".xdmf":
        file_name_base = file_name[:-5]

    else:
        if "." in os.path.basename(file_name):
            raise ValueError(f"Unrecognized file ending: {file_name}!")
        else:
            file_name_base = file_name

    # Make sure we have to absolute file name at hand.
    file_name_base = os.path.abspath(file_name_base)

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


def read_data(filebasename):

    print(f"Read data from {filebasename} ...")

    with SimulationFile(filebasename) as f:
        print(f["Function"].keys())

        num_particles = len(f["Function"].keys()) // 2

        print(f"Found {num_particles} particles.")

        # grid coordinates
        if "mesh" in f["Mesh"].keys():
            x_data = f["Mesh/mesh/geometry"][()]
        elif "Grid" in f["Mesh"].keys():
            x_data = f["Mesh/Grid/geometry"][()]
        else:
            raise ValueError("Neither 'Mesh' nor 'Grid' detected in " + filebasename)

        t_keys = f["Function/y_0"].keys()

        # time steps (convert from string to float)
        t = [float(t.replace("_", ".")) for t in t_keys]

        # list of data stored as numpy arrays
        u_data = np.array([
            [(f[f"Function/y_{i_part}"][u_key][()].squeeze(),
              f[f"Function/mu_{i_part}"][u_key][()].squeeze())
             for i_part in range(num_particles)] for u_key in t_keys])

    # It is necessary to sort the input by the time.
    sorted_indx = np.argsort(t)

    t = np.array(t)[sorted_indx]
    u_data = np.array(u_data)[sorted_indx]

    filebasename = strip_off_xdmf_file_ending(filebasename)

    # Read the runtime analysis output.
    rt_data = np.loadtxt(filebasename + "_rt.txt")

    # Total charge is not normalized.
    rt_data[:, 1] /= num_particles

    return num_particles, t, x_data, u_data, rt_data
