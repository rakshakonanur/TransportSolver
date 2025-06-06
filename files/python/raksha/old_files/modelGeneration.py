## Code to create cylinder models (.xdmf files) from 1D models (.vtp) generated by SVCCO

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

try:
    import vtk
except ImportError:
    raise ImportError("Could not find vtk, please install using pip or conda")

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

    


if __name__ == "__main__":
    # Hardcode the folder and save directory
    model_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches/3d_tmp"
    base_save_directory = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches/Results"

    process_all_models_in_directory(model_directory, base_save_directory)
