import argparse
from pathlib import Path
import dolfinx
import dolfinx.fem.petsc as petsc_fem
import numpy as np
import ufl
from dolfinx import common, default_scalar_type, fem, io, la, log
from mpi4py import MPI
from petsc4py import PETSc
import os
from create_and_convert_2D_mesh import markers
from dolfinx.io import gmshio, XDMFFile

has_tqdm = True
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    has_tqdm = False
    print("To view progress with progressbar please install tqdm: `pip3 install tqdm`")

log.set_log_level(log.LogLevel.ERROR)

def import_mesh(path_name):
    """
    Import a mesh and its corresponding facet tags from XDMF files.

    Args:
        path_mesh (str): Path to the XDMF file containing the mesh.
        path_facets (str): Path to the XDMF file containing the facet tags.

    Returns:
        dolfinx.cpp.mesh.Mesh: The imported mesh.
    """

    # Read the mesh from the XDMF file
    with XDMFFile(MPI.COMM_WORLD, path_name, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim, 
                                        domain.topology.dim - 1)

    return domain

def generate_facets(path_name: str, outdir: str):
    gdim = 3
    os.system(f"gmsh -{gdim} -optimize_netgen cfd.geo")

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh = import_mesh(path_name)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path_name, "r") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)

def IPCS(outdir: Path, dim: int, degree_u: int,
         jit_options: dict = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]}):
    assert degree_u >= 2

    mesh_dir = Path("meshes")
    if not mesh_dir.exists():
        raise RuntimeError(f"Could not find {str(mesh_dir)}")
    # Read in mesh
    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, f"meshes/channel{dim}D.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, tdim)
        mesh.topology.create_connectivity(fdim, tdim)

    with io.XDMFFile(comm, f"meshes/channel{dim}D_facets.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Facet tags")

    # Define function spaces
    V = fem.functionspace(mesh, ("Lagrange", degree_u, (mesh.geometry.dim, )))
    Q = fem.functionspace(mesh, ("Lagrange", degree_u - 1))

    # Temporal parameters
    t = 0
    dt = default_scalar_type(1e-2)
    T = 8

    # Physical parameters
    nu = 0.001
    f = fem.Constant(mesh, default_scalar_type((0,) * mesh.geometry.dim))
    H = 0.41
    Um = 2.25

    # Define functions for the variational form
    uh = fem.Function(V)
    uh.name = "Velocity"
    u_tent = fem.Function(V)
    u_tent.name = "Tentative_velocity"
    u_old = fem.Function(V)
    ph = fem.Function(Q)
    ph.name = "Pressure"
    phi = fem.Function(Q)
    phi.name = "Phi"

    # Define variational forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # ----Step 1: Tentative velocity step----
    w_time = fem.Constant(mesh, 3 / (2 * dt))
    w_diffusion = fem.Constant(mesh, default_scalar_type(nu))
    a_tent = w_time * ufl.inner(u, v) * dx + w_diffusion * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L_tent = (ufl.inner(ph, ufl.div(v)) + ufl.inner(f, v)) * dx
    L_tent += fem.Constant(mesh, 1 / (2 * dt)) * ufl.inner(4 * uh - u_old, v) * dx
    # BDF2 with implicit Adams-Bashforth
    bs = 2 * uh - u_old
    a_tent += ufl.inner(ufl.grad(u) * bs, v) * dx
    # Temam-device
    a_tent += 0.5 * ufl.div(bs) * ufl.inner(u, v) * dx

    # Find boundary facets and create boundary condition
    inlet_facets = mt.indices[mt.values == markers["Inlet"]]
    inlet_dofs = fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_facets = mt.indices[mt.values == markers["Walls"]]
    wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
    obstacle_facets = mt.indices[mt.values == markers["Obstacle"]]
    obstacle_dofs = fem.locate_dofs_topological(V, fdim, obstacle_facets)

    def inlet_velocity(t):
        if mesh.geometry.dim == 3:
            return lambda x: ((16 * np.sin(np.pi * t / T) * Um * x[1] * x[2] * (H - x[1]) * (H - x[2]) / (H**4),
                               np.zeros(x.shape[1]), np.zeros(x.shape[1])))
        elif mesh.geometry.dim == 2:
            U = 1.5 * np.sin(np.pi * t / T)
            return lambda x: np.row_stack((4 * U * x[1] * (0.41 - x[1]) / (0.41**2), np.zeros(x.shape[1])))

    u_inlet = fem.Function(V)
    u_inlet.interpolate(inlet_velocity(t))
    zero = np.array((0,) * mesh.geometry.dim, dtype=default_scalar_type)
    bcs_tent = [fem.dirichletbc(u_inlet, inlet_dofs), fem.dirichletbc(
        zero, wall_dofs, V), fem.dirichletbc(zero, obstacle_dofs, V)]
    a_tent = fem.form(a_tent, jit_options=jit_options)
    A_tent = petsc_fem.assemble_matrix(a_tent, bcs=bcs_tent)
    A_tent.assemble()
    L_tent = fem.form(L_tent, jit_options=jit_options)
    b_tent = fem.Function(V)

    # Step 2: Pressure correction step
    outlet_facets = mt.indices[mt.values == markers["Outlet"]]
    outlet_dofs = fem.locate_dofs_topological(Q, fdim, outlet_facets)
    bcs_corr = [fem.dirichletbc(default_scalar_type(0), outlet_dofs, Q)]
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)
    a_corr = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
    L_corr = - w_time * ufl.inner(ufl.div(u_tent), q) * dx
    a_corr = fem.form(a_corr, jit_options=jit_options)
    A_corr = petsc_fem.assemble_matrix(a_corr, bcs=bcs_corr)
    A_corr.assemble()

    b_corr = fem.Function(Q)
    L_corr = fem.form(L_corr, jit_options=jit_options)

    # Step 3: Velocity update
    a_up = fem.form(ufl.inner(u, v) * dx, jit_options=jit_options)
    L_up = fem.form((ufl.inner(u_tent, v) - w_time**(-1) * ufl.inner(ufl.grad(phi), v)) * dx,
                    jit_options=jit_options)
    A_up = petsc_fem.assemble_matrix(a_up)
    A_up.assemble()
    b_up = fem.Function(V)

    # Setup solvers
    rtol = 1e-8
    atol = 1e-8
    solver_tent = PETSc.KSP().create(comm)  # type: ignore
    solver_tent.setOperators(A_tent)
    solver_tent.setTolerances(rtol=rtol, atol=atol)
    solver_tent.rtol = rtol
    solver_tent.setType("bcgs")
    solver_tent.getPC().setType("jacobi")
    # solver_tent.setType("preonly")
    # solver_tent.getPC().setType("lu")
    # solver_tent.getPC().setFactorSolverType("mumps")

    solver_corr = PETSc.KSP().create(comm)  # type: ignore
    solver_corr.setOperators(A_corr)
    solver_corr.setTolerances(rtol=rtol, atol=atol)
    # solver_corr.setType("preonly")
    # solver_corr.getPC().setType("lu")
    # solver_corr.getPC().setFactorSolverType("mumps")
    solver_corr.setInitialGuessNonzero(True)
    solver_corr.max_it = 200
    solver_corr.setType("gmres")
    solver_corr.getPC().setType("hypre")
    solver_corr.getPC().setHYPREType("boomeramg")

    solver_up = PETSc.KSP().create(comm)  # type: ignore
    solver_up.setOperators(A_up)
    solver_up.setTolerances(rtol=rtol, atol=atol)
    # solver_up.setType("preonly")
    # solver_up.getPC().setType("lu")
    # solver_up.getPC().setFactorSolverType("mumps")
    solver_up.setInitialGuessNonzero(True)
    solver_up.max_it = 200
    solver_up.setType("cg")
    solver_up.getPC().setType("jacobi")

    # Create output files
    out_u = io.VTXWriter(comm, outdir / f"u_{dim}D.bp", [uh], engine="BP4")
    out_p = io.VTXWriter(comm, outdir / f"p_{dim}D.bp", [ph], engine="BP4")
    out_u.write(t)
    out_p.write(t)

    # Solve problem
    N = int(T / dt)
    if has_tqdm:
        time_range = tqdm(range(N))
    else:
        time_range = range(N)
    for i in time_range:

        t += dt
        # Solve step 1
        with common.Timer("~Step 1"):
            u_inlet.interpolate(inlet_velocity(t))
            A_tent.zeroEntries()
            petsc_fem.assemble_matrix(A_tent, a_tent, bcs=bcs_tent)  # type: ignore
            A_tent.assemble()

            b_tent.x.array[:] = 0
            petsc_fem.assemble_vector(b_tent.vector, L_tent)
            petsc_fem.apply_lifting(b_tent.vector, [a_tent], [bcs_tent])
            b_tent.x.scatter_reverse(la.InsertMode.add)
            petsc_fem.set_bc(b_tent.vector, bcs_tent)
            solver_tent.solve(b_tent.vector, u_tent.vector)
            u_tent.x.scatter_forward()

        # Solve step 2
        with common.Timer("~Step 2"):
            b_corr.x.array[:] = 0
            petsc_fem.assemble_vector(b_corr.vector, L_corr)
            petsc_fem.apply_lifting(b_corr.vector, [a_corr], [bcs_corr])
            b_corr.x.scatter_reverse(la.InsertMode.add)
            petsc_fem.set_bc(b_corr.vector, bcs_corr)
            solver_corr.solve(b_corr.vector, phi.vector)
            phi.x.scatter_forward()

            # Update p and previous u
            ph.vector.axpy(1.0, phi.vector)
            ph.x.scatter_forward()

            u_old.x.array[:] = uh.x.array
            u_old.x.scatter_forward()

        # Solve step 3
        with common.Timer("~Step 3"):
            b_up.x.array[:] = 0
            petsc_fem.assemble_vector(b_up.vector, L_up)
            b_up.x.scatter_reverse(la.InsertMode.add)
            solver_up.solve(b_up.vector, uh.vector)
            uh.x.scatter_forward()

        with common.Timer("~IO"):
            out_u.write(t)
            out_p.write(t)

    out_u.close()
    out_p.close()

    t_step_1 = MPI.COMM_WORLD.gather(common.timing("~Step 1"), root=0)
    t_step_2 = MPI.COMM_WORLD.gather(common.timing("~Step 2"), root=0)
    t_step_3 = MPI.COMM_WORLD.gather(common.timing("~Step 3"), root=0)
    io_time = MPI.COMM_WORLD.gather(common.timing("~IO"), root=0)
    if comm.rank == 0:
        print("Time-step breakdown")
        for i, step in enumerate([t_step_1, t_step_2, t_step_3]):
            step_arr = np.asarray(step)
            time_per_run = step_arr[:, 1] / step_arr[:, 0]
            print(f"Step {i+1}: Min time: {np.min(time_per_run):.3e}, Max time: {np.max(time_per_run):.3e}")
        io_time_arr = np.asarray(io_time)
        time_per_run = io_time_arr[:, 1] / io_time_arr[:, 0]
        print(f"IO {i+1}:   Min time: {np.min(time_per_run):.3e}, Max time: {np.max(time_per_run):.3e}")

    # common.list_timings(comm, [common.TimingType.wall])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run the DFG 2D-3 benchmark"
        + "http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--degree-u", default=2, type=int, dest="degree", help="Degree of velocity space")
    _2D = parser.add_mutually_exclusive_group(required=False)
    _2D.add_argument('--3D', dest='threed', action='store_true', help="Use 3D mesh", default=False)
    parser.add_argument("--outdir", default="results", type=str, dest="outdir", help="Name of output folder")
    args = parser.parse_args()
    dim = 3 if args.threed else 2
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    IPCS(outdir, dim=dim, degree_u=args.degree)