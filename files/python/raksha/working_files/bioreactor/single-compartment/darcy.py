import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
import os
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

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
import logging
logging.basicConfig(level=logging.DEBUG)

WALL = 0
OUTLET = 1

"""
    From the DOLFINx tutorial: Mixed formulation of the Poisson equation
    https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_mixed-poisson.html

    Weak imposition of Dirichlet boundary conditions using Nitsche's method:
    https://jsdokken.com/dolfinx-tutorial/chapter1/nitsche.html
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

def import_function(mesh, velocity_file, pressure_file):
    """
    Import a function from a BP file.
    """

    # Create function spaces
    P1 = dfx.fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1))
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)

    u_in = dfx.fem.Function(P1vec)
    p_in = dfx.fem.Function(P1)

    for i, timestamp in enumerate(adios4dolfinx.read_timestamps(velocity_file, comm=MPI.COMM_WORLD, function_name="f")):
        adios4dolfinx.read_function(velocity_file, u_in, time=timestamp, name="f")
        adios4dolfinx.read_function(pressure_file, p_in, time=timestamp, name="f")
        print(
            f"{MPI.COMM_WORLD.rank + 1}/{MPI.COMM_WORLD.size}: ",
            f"Function read in correctly at time {timestamp}",
        )

    

    return u_in, p_in    

def convert_vertex_tags_to_facet_tags(mesh, vertex_tags, u_vec):
    """
    Convert vertex-based tags (dim=0) to facet-based tags (dim=dim-1),
    and reduce u_vec to assign only one facet per tagged vertex.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)  # facet -> vertex
    mesh.topology.create_connectivity(fdim, tdim)  # facet -> cell

    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    vertex_to_facet = {}

    facet_indices = []
    facet_values = []
    total_facets, total_values, sorted_facets = [],[],[]
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
    
    total_facets.append(facet_indices)
    total_values.append(np.full_like(facet_indices, OUTLET))  # Tag 0 for internal facets
    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values = np.array(facet_values, dtype=np.int32)
    sorted_indices = np.argsort(facet_indices)
    internal_tags = dfx.mesh.meshtags(mesh, fdim, facet_indices[sorted_indices], facet_values[sorted_indices])

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
    print("Rescaled u_vec shape:", u_vec_rescaled.shape, flush=True)

    # Determine the wall facets
    wall_BC_indices, wall_BC_markers = [], []
    wall_BC_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    total_facets.append(wall_BC_facets)
    total_values.append(np.full_like(wall_BC_facets, WALL)) # Tag 0 for wall BCs
    wall_BC_indices.append(wall_BC_facets)
    wall_BC_markers.append(np.full_like(wall_BC_facets, WALL))  # Tag 0 for wall BCs

    wall_BC_indices = np.hstack(wall_BC_indices).astype(np.int32)  # Ensure facets are in int32 format
    wall_BC_markers = np.hstack(wall_BC_markers).astype(np.int32)  # Ensure markers are in int32 format
    sorted_facets = np.argsort(wall_BC_indices)
    external_tags = dfx.mesh.meshtags(mesh, fdim, wall_BC_indices[sorted_facets], wall_BC_markers[sorted_facets])
    print("External facets for wall BCs:", wall_BC_indices, flush=True)

    # Remove external facets from facet_indices and facet_values
    sorted_facets = np.argsort(facet_indices)
    velocity_facets = dfx.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_values[sorted_facets])
    facet_indices = np.setdiff1d(facet_indices, wall_BC_indices)
    facet_values = np.setdiff1d(facet_values, wall_BC_markers)
    print("Internal facets after removing wall BCs:", facet_indices, flush=True)

    # Save both meshtags for visualization
    total_values = np.hstack(total_values).astype(np.int32)
    total_facets = np.hstack(total_facets).astype(np.int32)
    sorted_facets = np.argsort(total_facets)
    print("Total facets:", total_facets[sorted_facets], flush=True)

    # Write mesh and tags to output files
    if mesh.comm.rank == 0:
        out_str = './output/mesh_tags.xdmf'
        with XDMFFile(mesh.comm, out_str, 'w') as xdmf_out:
            xdmf_out.write_mesh(mesh)
            mesh.topology.create_connectivity(fdim, tdim)
            xdmf_out.write_meshtags(
                dfx.mesh.meshtags(mesh, fdim, total_facets[sorted_facets], total_values[sorted_facets]),
                mesh.geometry
            )

    return internal_tags, external_tags, velocity_facets, u_vec_rescaled

def single_compartment(self, mesh, velocity_facets, M, W, p):
    Q_bio = 0.05  # given from Qterm in 1D sim

    # Calculate volume of the bioreactor
    DG = element("DG", mesh.basix_cell(), 0)
    v_ = dfx.fem.functionspace(mesh, DG)  
    vol = ufl.TestFunction(v_)
    volume_form = fem.form(vol * dx)  # Volume integral form
    V_bio = dfx.fem.assemble_scalar(volume_form)  # Assemble integral of v
    print(f"Mesh volume: {V_bio}", flush=True) # Volume of the bioreactor

    Pcap = 15  # Capillary pressure
    Psnk = 0  # Sink pressure

    # Spatial average of source pressure:
    fdim = mesh.topology.dim - 1  # Facet dimension
    outlet_facets = velocity_facets.find(OUTLET)  # Tag 1 is the outlets of the branched network
    print(f"Outlet facets: {outlet_facets}", flush=True)
    # dofs_branch = dfx.fem.locate_dofs_topological((M.sub(0), W), fdim, outlet_facets)
    Psrc = 60 # self.bc_outlet.x.array[dofs_branch]  # Get the values of the outlet BCs
    print(f"Outlet pressures: {Psrc}", flush=True)
    Psrc_avg = np.mean(Psrc)  # Average pressure at the outlets
    print(f"Average pressure at outlets: {Psrc_avg}", flush=True)

    beta_src = (Q_bio / V_bio) * (1/(Psrc_avg -Pcap)) # Source term coefficient
    beta_snk = (Q_bio / V_bio) * (1/(Pcap - Psnk))  # Sink term coefficient

    Psrc_c = dfx.fem.Constant(mesh, PETSc.ScalarType(Psrc))
    Psnk_c = dfx.fem.Constant(mesh, PETSc.ScalarType(Psnk))
    beta_src_c = dfx.fem.Constant(mesh, PETSc.ScalarType(beta_src))
    beta_snk_c = dfx.fem.Constant(mesh, PETSc.ScalarType(beta_snk))
    
    return beta_src_c * Psrc_c + beta_snk_c * Psnk_c, - p * (beta_src_c + beta_snk_c)

class PerfusionSolver:
    def __init__(self, mesh_tag_file: str, velocity_file: str, pressure_file: str):
        """
        Initialize the PerfusionSolver with a given STL file and branching data.
        """
        self.D_value = 1e-2
        self.element_degree = 1
        self.write_output = True
        # self.u_vec, self.p_out, self.outlet_coords = branch_mesh_tagging()
        self.mesh, self.vertex_tags = import_mesh(mesh_tag_file)
        self.velocity, self.pressure = import_function(self.mesh, velocity_file, pressure_file)
        self.internal_tags, self.external_tags, self.velocity_facets, self.u_vec = convert_vertex_tags_to_facet_tags(self.mesh, self.vertex_tags, self.u_vec)

        self.t = 0
        self.dt = 1  # Time step size, can be adjusted as needed
        self.T = 10  # Total simulation time, can be adjusted as needed

    def setup(self):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        
        # k = self.element_degree
        k = 1
        P_el = element("DG", self.mesh.basix_cell(), k-1)
        u_el = element("BDM", self.mesh.basix_cell(), k, shape=(self.mesh.geometry.dim,))
        M_el = mixed_element([P_el, u_el])

        # Define function spaces
        M = dfx.fem.functionspace(self.mesh, M_el) # Mixed function space
        W, _ = M.sub(0).collapse()  # Pressure function space
        V, _ = M.sub(1).collapse()  # Velocity function space
        (p, u) = ufl.TrialFunctions(M) # Trial functions for pressure and velocity
        (v, w) = ufl.TestFunctions(M) # Test functions for pressure and velocity

        dx = Measure("dx", self.mesh) # Cell integrals
        ds = Measure("ds", self.mesh, subdomain_data=self.external_tags) # External facet integrals
        dS = Measure("dS", domain=self.mesh, subdomain_data=self.internal_tags) # Internal facet integrals
        n = FacetNormal(self.mesh) # Normal vector to the facets

        kappa = 2e-5
        mu = 1
        kappa_over_mu = fem.Constant(self.mesh, dfx.default_scalar_type(kappa/mu))
        phi = fem.Constant(self.mesh, dfx.default_scalar_type(0.1)) # Porosity of the medium, ranging from 0 to 1
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        nitsche = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # Nitsche parameter

        # Velocity boundary conditions
        self.bc_velocity = dfx.fem.Function(V)
        self.bc_velocity.x.array[:] = 1.0  # Initialize to one
        values = self.u_vec[10] # assume last timestep for now
        print("Original values:", values, flush=True)
        flat_values = values.flatten()  # Shape: (78,)
        print("Flat values for velocity BCs:", flat_values, flush=True)

        outlet_facets = self.velocity_facets.find(OUTLET)  # Tag 1 is the outlets of the branched network
        dofs_branch = dfx.fem.locate_dofs_topological((M.sub(1), V), fdim, outlet_facets)
        print("Shape of dofs for branch outlets:", len(dofs_branch[0]), flush=True)

        for i, dof in enumerate(dofs_branch[0]):
            self.bc_velocity.x.array[dof] = flat_values[i]

        print("Nonzero values in velocity BC:", np.count_nonzero(self.bc_velocity.x.array), flush=True)
        bcs = [dfx.fem.dirichletbc(self.bc_velocity, dofs_branch, M.sub(1))]

        # Pressure boundary conditions
        # To do this, first find the cells adjacent to the outlet vertices, and then impose
        # the dirichlet BCs weakly using Nitsche's method.

        # Wall boundary conditions
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        self.bc_wall = dfx.fem.Function(W)
        self.bc_wall.x.array[:] = 1.0  # Set wall BC to zero

        # Pressure outlet boundary conditions
        self.bc_outlet = dfx.fem.Function(W)
        self.bc_outlet.x.array[:] = 0.0  # Initialize to zero, length = number of cells
        # Set the outlet pressure based on the 1D NS simulation
        pressure_outlets = self.internal_tags.find(OUTLET)

        facet_to_cell = self.mesh.topology.connectivity(fdim, self.mesh.topology.dim)
        adjacent_cells = []

        for facet in pressure_outlets:
            connected_cells = facet_to_cell.links(facet)
            adjacent_cells.extend(connected_cells)

        # Remove duplicates and convert to array
        adjacent_cells = np.unique(adjacent_cells)

        # Create geometry for cells
        cell_coords = dfx.mesh.compute_midpoints(self.mesh, self.mesh.topology.dim, np.array(adjacent_cells, dtype=np.int32))
        closest_cells = []

        for outlet_pt in self.outlet_coords:
            # Compute Euclidean distance from outlet point to each cell midpoint
            distances = np.linalg.norm(cell_coords - outlet_pt, axis=1)
            
            # Find index of closest cell
            closest_idx = np.argmin(distances)
            closest_cell = adjacent_cells[closest_idx]

            closest_cells.append(closest_cell)

        self.bc_outlet.x.array[closest_cells] = self.p_out[10, :] # Use last timestep for now
        print("Outlet pressures:", self.p_out[10, :], flush=True)

        fRHS, fLHS = single_compartment(self, self.mesh, self.velocity_facets, M, W, p) # Source and sink terms

        a = inner(u, w) * dx + inner(p, div(w)) * dx + inner(div(u), v) * dx + inner(fLHS, v) * dx
        L = -inner(fRHS, v) * dx

        # Impose Nitsche boundary conditions
        # Wall BC: might be working? commented out because it looks weird
        # more consistent way of enforcing- else can include only last term
        # a += (-dot(grad(p),n)*v - dot(grad(v),n)*p + nitsche /hf * p *v) * ds(WALL)
        # L += (-dot(grad(v),n)*self.bc_wall + nitsche /hf * self.bc_wall * v) * ds(WALL)

        # # Outlet BC
        a += (-dot(grad(p),n)*v - dot(grad(v),n)*p + nitsche /hf * p *v)("+") * dS(OUTLET)  
        a += (-dot(grad(p),n)*v - dot(grad(v),n)*p + nitsche /hf * p *v)("-") * dS(OUTLET)  

        L += (-dot(grad(v),n)*self.bc_outlet + nitsche /hf * self.bc_outlet * v)("+") * dS(OUTLET)  
        L += (-dot(grad(v),n)*self.bc_outlet + nitsche /hf * self.bc_outlet * v)("-") * dS(OUTLET)  

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

        vtkfile = VTKFile(MPI.COMM_WORLD, "u.vtu", "w")

        # Write the function to the VTK file
        vtkfile.write_function(u_interp)

if __name__ == "__main__":
    # Example usage
    mesh_tag_file = "../geometry/tagged_branches.xdmf"
    vel_file = "../geometry/velocity_checkpoint.bp"
    pressure_file = "../geometry/pressure_checkpoint.bp"
    solver = PerfusionSolver(mesh_tag_file, vel_file, pressure_file)
    solver.setup()
    print("PerfusionSolver setup complete.")

        



