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
        mesh_tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, mesh_tags

def remove_tags(mesh, meshtags):
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)  # facet → vertex
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)

    all_indices = meshtags.indices
    all_values = meshtags.values

    inlet = np.array([0.0, 0.41, 0.34])
    coords = mesh.geometry.x

    kept_indices = []
    kept_values = []

    for i, facet in enumerate(all_indices):
        vertex_ids = facet_to_vertex.links(facet)
        facet_coords = coords[vertex_ids]
        centroid = np.mean(facet_coords, axis=0)
        
        # If this facet's centroid is NOT the inlet coordinate, keep it
        if not np.allclose(centroid, inlet, atol=1e-6):
            kept_indices.append(facet)
            kept_values.append(all_values[i])

    new_meshtags = dfx.mesh.meshtags(mesh, fdim, np.array(kept_indices, dtype=np.int32), np.array(kept_values, dtype=np.int32))
    return new_meshtags


def import_function(velocity_file, pressure_file, other_mesh, internal_facets, external_facets):
    """
    Import a function from a BP file.
    """

    with XDMFFile(MPI.COMM_WORLD, "../geometry/branched_network.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    mesh = adios4dolfinx.read_mesh(filename = Path("../geometry/tagged_branches.bp"), comm=MPI.COMM_WORLD)
    old_meshtags = adios4dolfinx.read_meshtags(filename = Path("../geometry/tagged_branches.bp"), mesh=mesh, meshtag_name="mesh_tags")
    meshtags = remove_tags(mesh, old_meshtags)  # Remove tags with value 0

    # Split vertex tags into internal and external facets
    internal_1dfacets, external_1dfacets = split_vertex_tags_by_facet_tags(mesh, meshtags.indices, other_mesh, internal_facets, external_facets)

    # Create function spaces
    P1 = dfx.fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1))
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)

    total_velocity = [] # store velocity data from all timesteps
    total_pressure = [] # store pressure data from all timesteps

    u_in = dfx.fem.Function(P1vec)
    p_in = dfx.fem.Function(P1)
    print(adios4dolfinx.read_timestamps(pressure_file, comm=MPI.COMM_WORLD, function_name="f").shape)

    for timestamp in adios4dolfinx.read_timestamps(pressure_file, comm=MPI.COMM_WORLD, function_name="f"):
        adios4dolfinx.read_function(velocity_file, u_in, time=timestamp, name="f")
        adios4dolfinx.read_function(pressure_file, p_in, time=timestamp, name="f")
        total_velocity.append(u_in.x.array.copy())
        total_pressure.append(p_in.x.array.copy())
    
    # vertex_indices = meshtags.indices[internal_1dfacets]  # Get indices of vertices tagged with 2
    vertex_coords = mesh.geometry.x[internal_1dfacets]

    return total_velocity, total_pressure, vertex_coords, meshtags, internal_1dfacets, external_1dfacets


import numpy as np
from scipy.spatial import cKDTree

def split_vertex_tags_by_facet_tags(
    mesh, vertex_indices, other_mesh, other_mesh_internal_facets, other_mesh_external_facets):

    fdim = other_mesh.topology.dim - 1
    other_mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = other_mesh.topology.connectivity(fdim, 0)
    
    # Get vertices connected to internal facets on other mesh
    internal_vertices = set()
    for facet in other_mesh_internal_facets.indices:
        internal_vertices.update(facet_to_vertex.links(facet))
    internal_coords = other_mesh.geometry.x[list(internal_vertices), :]
    
    # Get vertices connected to external facets on other mesh
    external_vertices = set()
    for facet in other_mesh_external_facets.indices:
        external_vertices.update(facet_to_vertex.links(facet))
    external_coords = other_mesh.geometry.x[list(external_vertices), :]
    
    # Build KD-trees
    tree_internal = cKDTree(internal_coords)
    tree_external = cKDTree(external_coords)
    
    # Coordinates of input vertex_indices on target mesh
    coords = mesh.geometry.x[vertex_indices, :]
    
    # Query nearest distances
    dist_internal, _ = tree_internal.query(coords)
    dist_external, _ = tree_external.query(coords)
    
    # Classify indices by closer distance
    is_internal = dist_internal <= dist_external
    is_external = ~is_internal
    
    internal_indices = vertex_indices[is_internal]
    external_indices = vertex_indices[is_external]

    print(f"Internal indices: {internal_indices}, External indices: {external_indices}", flush=True)
    
    return internal_indices, external_indices

def separate_tags(mesh, meshtags):
    """
    Separate vertex tags into internal and external facets.
    """

    fdim = mesh.topology.dim - 1  # Facet dimension

    # External facets (on boundary)
    boundary_facets = dfx.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True)
    )
    boundary_facets_set = set(boundary_facets)

    facet_indices, facet_values = meshtags.indices[meshtags.values == 1], meshtags.values[meshtags.values == 1]
    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values  = np.array(facet_values, dtype=np.int32)

    # Split into external/internal facets
    is_boundary_facet = np.isin(facet_indices, boundary_facets)

    external_facet_indices = facet_indices[is_boundary_facet]
    external_facet_values  = facet_values[is_boundary_facet]

    internal_facet_indices = facet_indices[~is_boundary_facet]
    internal_facet_values  = facet_values[~is_boundary_facet]

    # Create MeshTags
    external_tags = dfx.mesh.meshtags(mesh, fdim, external_facet_indices, external_facet_values)
    internal_tags = dfx.mesh.meshtags(mesh, fdim, internal_facet_indices, internal_facet_values)

    print("External facets tagged:", external_facet_indices)
    print("Number of external facets:", len(external_facet_indices))
    print("Internal facets tagged:", internal_facet_indices)
    print("Number of internal facets:", len(internal_facet_indices))

    return internal_tags, external_tags


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
        self.mesh, self.mesh_tags = import_mesh(mesh_tag_file)
        self.internal_tags, self.external_tags = separate_tags(self.mesh, self.mesh_tags)
        self.velocity, self.pressure, self.outlet_coords, self.mesh1dtags, self.internal_1dtags, self.external_1dtags = import_function(velocity_file, pressure_file, self.mesh, self.internal_tags, self.external_tags)

        self.t = 0
        self.dt = 1  # Time step size, can be adjusted as needed
        self.T = len(self.velocity)  # Total simulation time, can be adjusted as needed

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
        self.bc_velocity.x.array[:] = 0.0  # Initialize to zero
        values = self.velocity[250] # assume last timestep for now
        print("Original values:", values, flush=True)
        flat_values = values.flatten()  # Shape: (78,)
        print("Flat values for velocity BCs:", flat_values, flush=True)

        outlet_facets = self.mesh_tags.find(OUTLET)  # Tag 1 is the outlets of the branched network
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
        self.bc_wall.x.array[:] = 0.0  # Set wall BC to zero

        # Pressure outlet boundary conditions
        self.bc_outlet = dfx.fem.Function(W)
        self.bc_outlet.x.array[:] = 0.0  # Initialize to zero, length = number of cells
        # Set the outlet pressure based on the 1D NS simulation
        pressure_outlets = self.internal_tags.find(OUTLET)
        print("Pressure outlet facets:", pressure_outlets, flush=True)

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

        print("Closest cells to outlet:", len(closest_cells), flush=True)
        facets_1d = self.internal_1dtags
        print("Facets on outlet:", facets_1d, flush=True)
    
        self.bc_outlet.x.array[closest_cells] = self.pressure[250][facets_1d] # Use last timestep for now
        print("Outlet pressures:", self.pressure[250][facets_1d], flush=True)

        fRHS, fLHS = single_compartment(self, self.mesh, self.mesh_tags, M, W, p) # Source and sink terms

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
    mesh_tag_file = "../geometry/mesh_tags.xdmf"
    vel_file = "../geometry/velocity_checkpoint.bp"
    pressure_file = "../geometry/pressure_checkpoint.bp"
    solver = PerfusionSolver(mesh_tag_file, vel_file, pressure_file)
    solver.setup()
    print("PerfusionSolver setup complete.")

        



