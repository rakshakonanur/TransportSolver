import meshio
import glob
import os
import re
import vtk
import numpy as np
from scipy.interpolate import griddata
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree, distance
from vtk import vtkXMLPolyDataReader
from scipy.cluster.hierarchy import fclusterdata
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI
import pandas as pd
from collections import defaultdict

class SimulationInputs:
    def __init__(self, **kwargs):
        self.inputs = {}
        self.inputs["input_directory"] = kwargs.get("input_directory", None)
        self.inputs["output_directory"] = kwargs.get("output_directory", None)

    def load_vtp(self):
        """
        Load VTP time-step files from the specified directory and extract data.
        Need time-dependent velocity data for the advection-diffusion equation.
        The function reads all VTP files in the specified directory, extracts point coordinates,
        point data, cell data, and field data, and stores them in a list of dictionaries.
        Each dictionary corresponds to a time step and contains the filename, coordinates,
        point data, cell data, and field data.  

        """

        directory = self.inputs["input_directory"]

        def extract_arrays(data_obj):
            return {
                data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                for i in range(data_obj.GetNumberOfArrays())
            }

        # Find all model files in the specified directory
        print("Searching for model files in directory:", directory)
        
        # Get a sorted list of all VTP files
        vtp_files = sorted(glob.glob(os.path.join(directory, "*.vtp")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
        all_data = []

        for file in vtp_files:
            print(f"Reading: {file}")
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(file)
            reader.Update()
            polydata = reader.GetOutput()

            def extract_arrays(data_obj):
                return {
                    data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                    for i in range(data_obj.GetNumberOfArrays())
                }
            
            # Extract point coordinates
            points = polydata.GetPoints()

            # Convert to NumPy array
            num_points = points.GetNumberOfPoints()

            timestep_data = {
                "filename": file,
                "coords": np.array([points.GetPoint(i) for i in range(num_points)]),
                "point_data": extract_arrays(polydata.GetPointData()),
                "cell_data": extract_arrays(polydata.GetCellData()),
                "field_data": extract_arrays(polydata.GetFieldData())
            }

            all_data.append(timestep_data)
        import pickle

        with open("data_series.pkl", "wb") as f:
            pickle.dump(all_data, f)
        self.inputs["data_series"] = all_data

    def centerlineVelocity(self):
        """
        Interpolate the velocity data from the VTP files onto the centerline of the reference model.
        Args:
            data_series (list): List of dictionaries containing the VTP data.
            ref_model (pyvista.PolyData): Reference model for interpolation.
        """
        path = self.inputs["output_directory"]
        data_series = self.inputs["data_series"]
        
        print("Number of timesteps:", len(data_series))
        # Iterate over each timestep in the data series
        self.centerlineVel = []  # initialize as a list
        self.centerlineFlow = []  # initialize as a list

        for i in range(len(data_series)):
            print(f"Processing timestep {i}")

            area_1d = data_series[i]["point_data"]["Area"]  # (N,)
            flowrate_1d = data_series[i]["point_data"]["Flowrate"]  # (N,)
            reynolds_1d = data_series[i]["point_data"]["Reynolds"]  # (N,)
            coords_1d = data_series[i]["coords"]  # (N, 3)
            
            velocity_1d = flowrate_1d/area_1d # calculate velocity from flowrate and area
            centerline = velocity_1d[::20] # save only one point for each cross-section
            centerlineFlowrate = flowrate_1d[::20] # save only one point for each cross-section
            self.centerlineFlow.append(centerlineFlowrate) # save the flowrate values for each timestep in each row
            self.centerlineVel.append(centerline)# save the velocity values for each timestep in each row
            centerlineCoords = coords_1d.reshape(-1, 20, 3).mean(axis=1)
            
            output = pv.PolyData(centerlineCoords)

            output["Velocity_Magnitude"] = centerline
            output.save(path + f"averagedVelocity_{i:04d}.vtp")

        self.centerlineVelocity = np.array(self.centerlineVelocity)
        print("✅ Projected flowrate values saved to 'averageVelocity.vtp'")

    def output_velocity_interp(self):
        """
        Interpolates scalar velocity from VTP centerlines onto a 1D mesh and writes XDMF time series using DOLFINx.
        """

        import dolfinx
        import numpy as np
        import pyvista as pv
        from dolfinx.fem import Function, functionspace
        from dolfinx.io import XDMFFile
        from basix.ufl import element
        from mpi4py import MPI
        from scipy.spatial import cKDTree
        import meshio

        path = self.inputs["output_directory"]
        data_series = self.inputs["data_series"]
        num_timesteps = len(data_series)
        dt = 0.1  # Adjust as needed

        # === Load DOLFINx-compatible mesh ===
        with XDMFFile(MPI.COMM_WORLD, "bifurcation.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        # Create scalar P1 function space
        V = functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1))
        velocity_func = Function(V)

        # === Get coordinates of mesh dofs ===
        dof_coords = V.tabulate_dof_coordinates()

        # === Write time series with FEniCSx ===
        with XDMFFile(MPI.COMM_WORLD, "velocity_timeseries.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)

            tree = None  # KDTree for nearest neighbor search
            for i in range(num_timesteps):
                time = i * dt
                vtp_file = os.path.join(path, f"averagedVelocity_{i:04d}.vtp")
                print(f"Processing timestep {i}: {vtp_file}", flush=True)

                # --- Load VTP and extract data ---
                vtp = pv.read(vtp_file)
                vtp_points = vtp.points
                scalar_velocity = vtp.point_data["Velocity_Magnitude"]

                if tree is None:
                    tree = cKDTree(vtp_points)

                # Nearest neighbor interpolation
                distances, indices = tree.query(dof_coords)
                interpolated = scalar_velocity[indices]
                # interpolated = np.nan_to_num(interpolated)  # Avoid NaNs

                # Assign to DOLFINx function
                velocity_func.x.array[:] = interpolated
                velocity_func.x.scatter_forward()

                # Write to file
                xdmf.write_function(velocity_func, time)
    
    def create_directory(self): # create directory for output files
        """Creates a directory if it doesn't exist."""
        dir = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/Output"
        path = self.inputs["input_directory"]
        directory = Path(path)
        fileName = directory.parent.name
        current_time = datetime.now()
        date = f"{str(current_time.month).zfill(2)}{str(current_time.day).zfill(2)}{current_time.year%2000}"

        # Current path
        directory_path = dir +  '/' + str(date) 

        count = 0
        if os.path.exists(directory_path):
            for entry in os.scandir(directory_path):
                if entry.is_dir():
                    count += 1
        
        path_create = directory_path + '/' + 'Run' + str(count+1) + '_' + fileName + '/'

        os.makedirs(path_create)
        self.inputs["output_directory"] = path_create

if __name__ == "__main__":
    path="/mnt/c/Users/rkona/Documents/syntheticVasculature/1D Output/052125/Run4_100branches"
    setup= SimulationInputs(input_directory=path)
    setup.create_directory()
    setup.load_vtp()
    # ref_model = pv.read(path + "/1D Input Files/1d_model.vtp")
    setup.centerlineVelocity()
    setup.output_velocity_interp()
    # output_path = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/Output/042425/Run4_042425"

