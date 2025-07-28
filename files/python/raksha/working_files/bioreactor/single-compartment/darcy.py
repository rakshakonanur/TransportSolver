import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
import os
import pyvista as pv

from scipy.spatial     import cKDTree
from ufl               import avg, jump, dot, grad
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx import (fem, io, mesh)
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting, LinearProblem, set_bc
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from typing import List, Optional
from dolfinx.io import XDMFFile, VTKFile
import adios4dolfinx

"""
    From the DOLFINx tutorial: Mixed formulation of the Poisson equation
    https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_mixed-poisson.html
"""

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def import_mesh(xdmf_file):
    """
    Import a mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        vertex_tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, vertex_tags

def outlet_velocity_tagging(mesh):
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    boundary_facets_indices = dfx.mesh.exterior_facet_indices(mesh.topology)
    inlet = np.array([0,0.41,0.34]) #updated with units
    tol = 1e-6

    def near_inlet(x):
        return np.isclose(x[0],inlet[0]) & np.isclose(x[1],inlet[1]) & np.isclose(x[2],inlet[2])

    inlet_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, near_inlet)

    # Extract outlet facets: boundary facets excluding inlet facets
    outlet_facets = np.setdiff1d(boundary_facets_indices, inlet_facets)

    # inlet_facets = [boundary_facets_indices[0]] # first element corresponds to inlet
    # outlet_facets = boundary_facets_indices[1:] # all other elements correspond to outlets

    facet_indices = np.concatenate([inlet_facets, outlet_facets])
    facet_markers = np.concatenate([np.full(len(inlet_facets), 1, dtype=np.int32),
                                    np.full(len(outlet_facets), 2, dtype=np.int32)])
    facet_tag = dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_markers)
    print("Facet indices: ", facet_indices, flush=True)
    print("Facet markers: ", facet_markers, flush=True)
    return inlet_facets, outlet_facets, facet_tag

def branch_mesh_tagging():

    xdmf_file = "../../test_full_model/bifurcation.xdmf"
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    _, branch_outlet_facets, branch_facet_tag = outlet_velocity_tagging(mesh)
    u_val = vector_velocity(mesh, branch_outlet_facets, branch_facet_tag)
    return u_val

def vector_velocity(mesh, outlet_facets, facet_tag):
    
    # Load the averagedVelocity files for all time steps
    tree = None  # KDTree for nearest neighbor search
    num_timesteps = 11  # hard-coded for now, can be replaced with dynamic loading
    dt = 1
    output_path = "/Users/rakshakonanur/Documents/Research/Transport_Solver/TransportSolver/files/python/raksha/working_files/test_full_model/output"
    
    fdim = mesh.topology.dim - 1

    # Create scalar P1 function space
    DG = element("DG", mesh.basix_cell(), 1)
    V = dfx.fem.functionspace(mesh, DG)
    # u_val = Function(V)

    # === Get coordinates of mesh dofs ===
    dof_coords = V.tabulate_dof_coordinates()
    outlet_coords = dof_coords[facet_tag.find(2)]  # Coordinates of outlet facets
    num_dofs = len(outlet_facets)

    # Allocate u_val as a 2D array for all timesteps and dofs
    u_val = np.zeros((num_timesteps, num_dofs))

    # Step 1: Map outlet facets to their adjacent cells (1 per facet in DG)
    facet_to_cell = mesh.topology.connectivity(fdim, mesh.topology.dim)
    outlet_cells = np.array([facet_to_cell.links(f)[0] for f in outlet_facets], dtype=np.int32)

    # Step 2: Get tangents for these cells
    _, tangents = compute_element_tangents()  # tangents.shape == (num_cells, 3)
    outlet_tangents = tangents[outlet_cells]  # shape: (num_outlet_facets, 3)

    for i in range(num_timesteps):
        time = i * dt
        vtp_file = os.path.join(output_path + f"/averagedVelocity_{i:04d}.vtp")
        print(f"Processing timestep {i}: {vtp_file}", flush=True)

        # --- Load VTP and extract data ---
        vtp = pv.read(vtp_file)
        vtp_points = vtp.points
        scalar_velocity = vtp.point_data["Velocity_Magnitude"]
        

        if tree is None:
            tree = cKDTree(vtp_points)

        # Nearest neighbor interpolation
        distances, indices = tree.query(outlet_coords)
        interpolated = scalar_velocity[indices]
        # interpolated = np.nan_to_num(interpolated)  # Avoid NaNs

        u_val[i,:] = interpolated
        print(f"Interpolated values for timestep {i}: {interpolated}", flush=True)

    # Step 3: Multiply scalar velocity magnitudes by tangents to get vector velocity
    # u_val.shape = (num_timesteps, num_outlet_facets)
    # vector_u_val.shape = (num_timesteps, num_outlet_facets, 3)
    vector_u_val = u_val[:, :, np.newaxis] * outlet_tangents[np.newaxis, :, :]

    print("Vector-valued outlet velocities shape:", vector_u_val.shape)

    return vector_u_val

def compute_element_tangents():
        """Compute unit tangent vectors for each cell in a 1D mesh."""
        import numpy.linalg as la

        xdmf_file = "../../test_full_model/bifurcation.xdmf"
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

        mesh = mesh
        dim = mesh.topology.dim

        # Get cell indices and connectivity
        cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
        conn = mesh.topology.connectivity(dim, 0)
        cell_nodes = [conn.links(i) for i in cells]

        # Access node coordinates
        coords = mesh.geometry.x

        tangents = []
        for nodes in cell_nodes:
            p0 = coords[nodes[0]]
            p1 = coords[nodes[1]]
            delta = p1 - p0
            tangent = delta / la.norm(delta)
            tangents.append(tangent)
            
        return cells, np.array(tangents)       

def convert_vertex_tags_to_facet_tags(mesh, vertex_tags, u_vec):
    """
    Convert vertex-based tags (dim=0) to facet-based tags (dim=dim-1),
    and reduce u_vec to assign only one facet per tagged vertex.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)  # facet -> vertex

    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    vertex_to_facet = {}

    facet_indices = []
    facet_values = []
    used_facets = set()
    used_vertices = set()

    for facet in range(mesh.topology.index_map(fdim).size_local):
        vertex_ids = facet_to_vertex.links(facet)
        tags_on_vertices = vertex_tags.values[np.isin(vertex_tags.indices, vertex_ids)]

        if len(tags_on_vertices) > 0:
            # Assign tag only if vertex hasn't been used
            for v_id in vertex_ids:
                if v_id in vertex_tags.indices and v_id not in used_vertices:
                    tag = np.bincount(tags_on_vertices).argmax()
                    facet_indices.append(facet)
                    facet_values.append(tag)
                    used_vertices.add(v_id)
                    used_facets.add(facet)
                    break  # Only one facet per vertex

    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values = np.array(facet_values, dtype=np.int32)

    # Rescale u_vec: keep only the rows corresponding to used_facets
    used_facets = np.array(sorted(used_facets), dtype=np.int32)
    # Build mapping from facet index → index in u_vec
    facet_map = {facet: i for i, facet in enumerate(facet_indices)}
    u_vec_rescaled = []

    for facet in used_facets:
        i = facet_map[facet]  # Get index into u_vec
        u_vec_rescaled.append(u_vec[:, i, :])  # shape (timesteps, 1, 3)

    # Stack to get shape: (timesteps, num_used_facets, 3)
    u_vec_rescaled = np.stack(u_vec_rescaled, axis=1)


    # Write mesh and tags to output files
    if mesh.comm.rank == 0:
        out_str = './output/mesh_tags.xdmf'
        with XDMFFile(mesh.comm, out_str, 'w') as xdmf_out:
            xdmf_out.write_mesh(mesh)
            mesh.topology.create_connectivity(fdim, tdim)
            xdmf_out.write_meshtags(
                dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_values),
                mesh.geometry
            )

    return dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_values), u_vec_rescaled


class PerfusionSolver:
    def __init__(self, xdmf_file: str):
        """
        Initialize the PerfusionSolver with a given STL file and branching data.
        """
        self.D_value = 1e-2
        self.element_degree = 1
        self.write_output = True
        self.u_vec = branch_mesh_tagging()
        self.mesh, self.vertex_tags = import_mesh(xdmf_file)
        self.facets, self.u_vec = convert_vertex_tags_to_facet_tags(self.mesh, self.vertex_tags, self.u_vec)
        

        # self.u_val = outlet_velocity_tagging(self.mesh)  # Call to tag outlets and inlet
        self.t = 0
        self.dt = 1  # Time step size, can be adjusted as needed
        self.T = 10  # Total simulation time, can be adjusted as needed

    def setup(self):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        
        # k = self.element_degree
        k = 1
        P_el = element("Lagrange", self.mesh.basix_cell(), k)
        u_el = element("DG", self.mesh.basix_cell(), k-1, shape=(self.mesh.geometry.dim,))
        M_el = mixed_element([P_el, u_el])

        # Define function spaces
        M = dfx.fem.functionspace(self.mesh, M_el) # Mixed function space
        W, _ = M.sub(0).collapse()  # Pressure function space
        V, _ = M.sub(1).collapse()  # Velocity function space
        (p, u) = ufl.TrialFunctions(M) # Trial functions for pressure and velocity
        (v, w) = ufl.TestFunctions(M) # Test functions for pressure and velocity

        kappa = 1
        mu = 1
        kappa_over_mu = fem.Constant(self.mesh, dfx.default_scalar_type(kappa/mu))
        phi = fem.Constant(self.mesh, dfx.default_scalar_type(0.1)) # Porosity of the medium, ranging from 0 to 1
        f = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term
        beta  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10.0)) # Nitsche penalty paramete
        hf = ufl.CellDiameter(self.mesh) # Cell diamete

        # # Velocity boundary conditions
        # self.bc_outlet = dfx.fem.Function(V)
        # self.bc_outlet.x.array[:] = 1.0  # Initialize to one
        # # Block size (e.g., 3 for 3D velocity)
        # bs = self.bc_outlet.function_space.dofmap.index_map_bs
        # index_map = self.bc_outlet.function_space.dofmap.index_map
        # values = self.u_vec[10]
        # outlet_facets = self.facets.find(1)  # Tag 1 is the outlets of the branched network
        # dofs_branch = dfx.fem.locate_dofs_topological(V, fdim, outlet_facets)
        # print("Dofs branch: ", dofs_branch, flush=True)

        # # Loop over each DOF in the list
        # for i, dof in enumerate(dofs_branch):
        #     for j in range(bs):
        #         local_dof = dof * bs + j
        #         global_dof = index_map.local_to_global(np.array([local_dof], dtype=np.int32))[0]
        #         self.bc_outlet.x.array[global_dof] = values[i, j]

        # Pressure boundary conditions
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        bc_wall = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        wall_BC_facets = dfx.mesh.exterior_facet_indices(self.mesh.topology)
        wall_BC_dofs = dfx.fem.locate_dofs_topological((M.sub(0), W), fdim, wall_BC_facets)

        self.bc_func = dfx.fem.Function(W)
        self.bc_func.x.array[:] = 60.0  # * 1333.22  # Convert mmHg to Pa
        dofs = self.facets.find(1) # Tag 1 is the outlets of the branched network
        bcs = [dfx.fem.dirichletbc(self.bc_func, dofs),
               dfx.fem.dirichletbc(bc_wall, np.setdiff1d(wall_BC_dofs, dofs), M.sub(0))]
        
        dx = Measure("dx", self.mesh)
        ds = Measure("ds", self.mesh, subdomain_data=self.facets)
        a = inner(u, w) * dx + inner(p, div(w)) * dx + inner(div(u), v) * dx
        # a += beta/ hf * inner(u,w) * ds(1)
        L = -inner(f, v) * dx  
        # L += beta / hf * inner(self.bc_outlet, w) * ds(1)

        self.a = a
        self.L = L

        # Apply Dirichlet BCs
        self.bcs = bcs
        problem = LinearProblem(self.a, self.L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

        try:
            w_h = problem.solve()
        except PETSc.Error as e:  # type: ignore
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

        p_h, u_h = w_h.split()

        with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/p.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            file.write_function(p_h)

        with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/u.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            P1 = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
            u_interp = fem.Function(fem.functionspace(self.mesh, P1))

            # Interpolate the data
            u_interp.interpolate(u_h)

            # Write interpolated function
            file.write_function(u_interp)


if __name__ == "__main__":
    # Example usage
    xdmf_file = "../geometry/vertex_tags_nearest.xdmf"
    solver = PerfusionSolver(xdmf_file)
    solver.setup()
    print("PerfusionSolver setup complete.")

        



