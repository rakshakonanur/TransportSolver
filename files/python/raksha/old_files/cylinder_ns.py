# Create the flow-field results for Navier-Stokes equations

import pandas as pd
import vtk
import numpy as np
from dolfinx import fem, default_scalar_type, log
from dolfinx.fem import functionspace
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.io import XDMFFile, VTKFile
import ufl
from basix.ufl import element
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import os
import argparse
import time
import glob
from model_to_mesh import * # Convert output of code in vtp to xdmf
import logging
logger = logging.getLogger(__name__)

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

def process_all_models_in_directory(directory, base_save_dir):

    """
    Batch process all model files in a specified directory.

    Args: 
        directory (str): Path to the directory containing model files.
        base_save_dir (str): Base directory to save the processed models.

    Returns:
        None
    """

    # Find all model files in the specified directory
    print("Searching for model files in directory:", directory)
    vtp_files = glob.glob(os.path.join(directory, "*.vtp"))
    xdmf_files = glob.glob(os.path.join(directory, "*.xdmf"))
    stl_files = glob.glob(os.path.join(directory, "*.stl"))
    # csv_files = glob.glob(os.path.join(directory, "*.csv"))

    model_files = vtp_files + xdmf_files + stl_files 

    for model_file in model_files:
        # Extract the model name without the extension
        model_name = os.path.splitext(os.path.basename(model_file))[0]

        # Create a subdirectory for each model
        model_save_dir = os.path.join(base_save_dir, model_name)
        model_save_dir = base_save_dir+"/"+model_name+"/"+model_name
        os.makedirs(model_save_dir, exist_ok=True)

        # # Save the radius of the smallest vessel in the model directory
        # df = pd.read_csv(directory+"/"+model_name+".csv")
        
        # # Extract the minimum value from column 'W'
        # minRadius = df['Radius'].min()
        # print(f"Minimum radius for {model_name}: {minRadius}")
       
        minRadius = 0.008 # Hardcoded for now

        print(f"Processing model: {model_file}")
        main_(model_file, model_save_dir, minRadius)

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

def solve_ns(domain):

    """
    Solve the Navier-Stokes equations for the given mesh domain.

    Args:
        domain: The mesh domain (xdmf file).

    Returns:
        None
    """
    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)

    # Define function spaces
    v_cg2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim, ))
    s_cg1 = element("Lagrange", domain.topology.cell_name(), 1)
    V = functionspace(domain, v_cg2) # function space for velocity
    Q = functionspace(domain, s_cg1) # function space for pressure

    # Time parameters (hard-coded for now)
    t = 0
    T = 10
    num_steps = 100
    dt = T / num_steps

    # Create the trial and test functions for the velocity and pressure
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    # Get boundary facets and corresponding DOFs
    boundary_facets = dolfinx.cpp.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1,
                                                boundary_facets)
    hmax, _, _ = edge_max(domain)

def main_(model_path, save_dir, minRadius):

    if model_path.endswith(".xdmf"):
        domain = import_mesh(model_path)
        mesh_time = None

    elif model_path.endswith(".vtp") or model_path.endswith(".stl"):
        start_time = time.time()
        main(model_path, save_dir, minRadius)
        end_time = time.time()
        mesh_time = end_time - start_time
        domain = import_mesh(save_dir+".xdmf")

    edgemax, _, edgeavg = edge_max(domain)

    # Solve for the flow field using Navier-Stokes equations


if __name__ == "__main__":
    # Hardcode the folder and save directory
    model_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusion/files/python/raksha/meshInput"
    base_save_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusion/files/python/raksha/meshOutput"

    process_all_models_in_directory(model_directory, base_save_directory)

