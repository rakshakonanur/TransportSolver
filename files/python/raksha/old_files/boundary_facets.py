try:
    import vtk
except ImportError:
    raise ImportError("Could not find vtk, please install using pip or conda")

try:
    import dolfinx
except ImportError:
    raise ImportError("Could not find dolfinx, please install")

import numpy as np
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.io import XDMFFile, VTKFile
import ufl
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import os
import argparse
from model_to_mesh import *


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

    print(type(domain))
    print(type(domain.topology))

    return domain

def edge_max(domain):
    """
    Calculate the maximum, minimum, and average edge lengths of the cells in
    the mesh.

    Args:
        domain: The mesh domain (xdmf file).

    Returns:
        tuple: A tuple containing the maximum edge length, minimum edge length,
        and average edge length.
    """

    # Get the total number of cells in the mesh
    num_cells = domain.topology.index_map(domain.topology.dim).size_global

    # Create an array of cell indices
    cells = np.arange(num_cells, dtype=np.int32)

    # Create a new mesh object with the same topology and geometry
    domain = dolfinx.cpp.mesh.Mesh_float64(domain.comm, domain.topology,
                                           domain.geometry)

    # Calculate the edge lengths of the cells
    edge = dolfinx.cpp.mesh.h(domain, domain.topology.dim-2, cells)

    # Calculate the average, max and min edge length
    edge_avg = np.mean(edge)
    edge_max = max(edge)
    edge_min = min(edge)

    return edge_max, edge_min, edge_avg


def set_boundary_facets(domain, output_file="boundary_data.txt"):

    # Create a function space on the domain
    V = functionspace(domain, ("Lagrange", 1))

    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    # Get boundary facets and corresponding DOFs
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                boundary_facets)
    
    # Save facets, dof to file
    with open(output_file, "w") as f:
        f.write("Boundary Facets:\n")
        f.write(f"{boundary_facets.tolist()}\n")
        f.write("Boundary DOFs:\n")
        f.write(f"{boundary_dofs.tolist()}\n")

    print(f"Boundary data saved to {output_file}")

import glob

def process_all_models_in_directory(directory, base_save_dir):
    # Find all model files in the specified directory
    vtp_files = glob.glob(os.path.join(directory, "*.vtp"))
    xdmf_files = glob.glob(os.path.join(directory, "*.xdmf"))
    stl_files = glob.glob(os.path.join(directory, "*.stl"))

    model_files = vtp_files + xdmf_files + stl_files

    for model_file in model_files:
        # Extract the model name without the extension
        model_name = os.path.splitext(os.path.basename(model_file))[0]

        # Create a subdirectory for each model
        # model_save_dir = os.path.join(base_save_dir, model_name)
        model_save_dir = base_save_dir+"/"+model_name+"/"+model_name
        os.makedirs(model_save_dir, exist_ok=True)

        print(f"Processing model: {model_file}")
        main_(model_file, model_save_dir)

import time
# import pyvista as pv

def main_(model_path, save_dir):

    if model_path.endswith(".xdmf"):
        domain = import_mesh(model_path)
        mesh_time = None
        print("Imported mesh from XDMF file.")

    # edgemax, _, edgeavg = edge_max(domain)
    set_boundary_facets(domain, output_file="boundary_data.txt")

if __name__ == "__main__":
    # Hardcode the folder and save directory
    model_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusion/files/python/raksha/test_mesh"
    base_save_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusion/files/python/raksha/test_mesh"

    process_all_models_in_directory(model_directory, base_save_directory)

    


