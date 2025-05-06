# import ufl
import numpy   as np
# import dolfinx as dfx

# from ufl               import avg, jump, dot, grad
# from sys               import argv
# from mpi4py            import MPI
# from pathlib           import Path
# from petsc4py          import PETSc
# from basix.ufl         import element
# from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting

# print = PETSc.Sys.Print

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

class simulationInputs:

    def __init__(self, **kwargs):
        """
        Initialize the simulationInputs class with optional keyword arguments.
        """
        self.inputs = {}
        self.inputs["input_directory"] = kwargs.get("input_directory", None)
        self.inputs["output_directory"] = kwargs.get("output_directory", None)


    def load_pvd(self):
        
        directory = self.inputs["input_directory"]
        # Find all model files in the specified directory
        print("Searching for model files in directory:", directory)
        
        # Get a sorted list of all VTP files
        pvd_files = sorted(glob.glob(os.path.join(directory, "*.pvd")), key=lambda x: int(re.findall(r'\d+', x)[-1]))

        for file in pvd_files:
            # Parse the PVD file
            tree = ET.parse(file)
            root = tree.getroot()

            # Folder containing the PVD file and referenced datasets
            base_dir = os.path.dirname(file)

            all_data = []

            for dataset in root.iter("DataSet"):
                timestep = float(dataset.attrib["timestep"])
                file_path = os.path.join(base_dir, dataset.attrib["file"])
                print(f"Loading timestep {timestep}: {file_path}")

                # Guess reader based on file extension
                if file_path.endswith(".vtu"):
                    reader = vtk.vtkXMLUnstructuredGridReader()
                elif file_path.endswith(".vtp"):
                    reader = vtk.vtkXMLPolyDataReader()
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")

                reader.SetFileName(file_path)
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

            return all_data
        
    def load_vtp(self):

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
        self.inputs["data_series"] = all_data

    
    
    def import_mesh(self, path_name):
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

    def export_xdmf(self, path_export, mesh, function):
        """
        Export a mesh and its associated fenicsx function to an XDMF file.

        Args:
            path_export (str): Path to the XDMF file where the mesh and function
            will be saved.
            mesh: The mesh to be exported.
            function: The function associated with the mesh to be exported.

        """

        # Save the mesh and function in XDMF format for visualization
        with XDMFFile(MPI.COMM_WORLD, path_export, "w") as file:
            file.write_mesh(mesh)
            file.write_function(function)

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
            output["AverageVelocity"] = centerline
            output.save(path + f"averagedVelocity_{i:04d}.vtp")

        self.centerlineVelocity = np.array(self.centerlineVelocity)
        print("✅ Projected flowrate values saved to 'averageVelocity.vtp'")


    def load_vtp_points(self, filepath):
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        polydata = reader.GetOutput()
        return vtk_to_numpy(polydata.GetPoints().GetData())

    def detect_bifurcation(self):

        """
        Detect bifurcations using the bifurcationID from the centerline model.
        (path + "/1D Input Files/1d_model.vtp", path + "/1D Input Files/centerlines.vtp",output_path+"/averagedVelocity_0001.vtp", output_path)
        """
        def load_vtp(filepath):
            reader = vtkXMLPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            polydata = reader.GetOutput()
            return polydata
        
        def extract_arrays(data_obj):
            return {
                data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                for i in range(data_obj.GetNumberOfArrays())
            }
        
        path = self.inputs["input_directory"]
        output_path = self.inputs["output_directory"]
        model_1d = path + "/1D Input Files/1d_model.vtp"
        centerline_model = path + "/1D Input Files/centerlines.vtp"
        model = output_path + "/averagedVelocity_0001.vtp"


        polydata = load_vtp(centerline_model)
        self.point_data = extract_arrays(polydata.GetPointData())
        points = polydata.GetPoints()
        bifurcationCoords = []
        bifurcation = 0
        add_data = False

        for i in range(points.GetNumberOfPoints()):
            if self.point_data["BifurcationId"][i] == bifurcation:
                add_data = True
                bifurcationCoords.append(points.GetPoint(i))
            else:
                add_data = False
                if len(bifurcationCoords) > 0:
                    if bifurcation == 0:
                        merged_bifurcations = np.array(bifurcationCoords).mean(axis=0)
                    else:
                        merged_bifurcations = np.vstack([merged_bifurcations, np.array(bifurcationCoords).mean(axis=0)])
                    # print("Bifurcation coordinates:", bifurcationCoords)
                    bifurcationCoords = []
                    bifurcation += 1

        print("Bifurcation coordinates:", merged_bifurcations)
        np.save(output_path + "/bifurcationCoords.npy", merged_bifurcations)
        self.determine_segment(model_1d, model, output_path)
    
    def combine_bifurcations(self, bifurcationCoords, threshold=0.01):
        """
        Combine bifurcations into a single coordinate based off a distance threshold.
        """
        points = np.array(bifurcationCoords)
        
        # Cluster points based on distance threshold
        labels = fclusterdata(points, t=threshold, criterion='distance')

        # Average points in each cluster
        merged_points = []
        for label in np.unique(labels):
            cluster = points[labels == label]
            avg = cluster.mean(axis=0)
            merged_points.append(avg)

        merged_points = np.array(merged_points)
        print("Merged coordinates: ", merged_points)

    def read_vtp_points(self, filepath):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            return reader.GetOutput()
        
    def determine_segment(self, model_1d, model, output_dir):
        """
        Determine the segment of the bifurcation coordinates in the model.
        """
        

        def get_array(polydata, name):
            return vtk.util.numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray(name))

        def fit_line_and_classify(branch_id, inlet, outlet, test_points, tolerance=1e-2):
            """Return a boolean mask for test_points that lie on the branch line."""
            inlet = np.array(inlet)
            outlet = np.array(outlet)
            direction = outlet - inlet
            direction /= np.linalg.norm(direction)

            # Project each test point onto the line and compute perpendicular distance
            vecs = test_points - inlet
            projections = np.dot(vecs, direction)
            closest_pts = np.outer(projections, direction) + inlet
            dists = np.linalg.norm(test_points - closest_pts, axis=1)
            
            on_line = dists < tolerance
            return on_line

        # === Load vessel with branches ===
        branches_polydata = self.read_vtp_points(model_1d)
        points = np.array([branches_polydata.GetPoint(i) for i in range(branches_polydata.GetNumberOfPoints())])
        branch_ids = get_array(branches_polydata, "BranchId")
        paths = get_array(branches_polydata, "Path")

        # === Load the other .vtp file with unknown points ===
        test_polydata = self.read_vtp_points(model)
        test_points = np.array([test_polydata.GetPoint(i) for i in range(test_polydata.GetNumberOfPoints())])

        # === Create output label array ===
        labels = np.full(test_points.shape[0], -1, dtype=int)  # -1 means unassigned

        # === Process each branch ===
        unique_branches = np.unique(branch_ids)
        print(f"Processing {len(unique_branches)} branches...")

        for bid in unique_branches:
            indices = np.where(branch_ids == bid)[0]
            branch_points = points[indices]
            branch_paths = paths[indices]

            if np.sum(branch_paths == 0) == 0 or np.sum(branch_paths != 0) == 0:
                continue  # Need both inlet and outlet

            inlet = branch_points[branch_paths == 0][0]
            outlet = branch_points[branch_paths != 0][-1]  # Take last for stability

            # Find which test points lie on this branch
            mask = fit_line_and_classify(bid, inlet, outlet, test_points, tolerance=1e-2)
            labels[mask] = bid

        # === Save labeled VTP at each time step ===

        vtp_files = sorted(glob.glob(os.path.join(output_dir, "averaged*")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
        all_data = []

        for file in vtp_files:
            polydata = self.read_vtp_points(file)
            output = vtk.vtkPolyData()
            output.DeepCopy(polydata)

            label_array = vtk.util.numpy_support.numpy_to_vtk(labels, deep=True)
            label_array.SetName("SegmentLabel")
            output.GetPointData().AddArray(label_array)

            i =  int(re.findall(r'\d+', os.path.basename(file))[0])  # ➜ extract the timestep number from the filename

            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(output_dir+f"/labeled_segments_{i:04d}.vtp")
            writer.SetInputData(output)
            writer.Write()

        print("✅ Segmentation complete. Saved to 'labeled_segments.vtp'")



    def create_sim_inputs(self):
        """
        Create simulation inputs for the advection-diffusion model.
        """
        def get_array(polydata, name):
            return vtk.util.numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray(name))

        # Load the mesh and facet tags
        path = self.inputs["input_directory"]
        model_1d = path + "/1D Input Files/1d_model.vtp"
        polydata = self.read_vtp_points(model_1d)

        branch_ids = get_array(polydata, "BranchId")
        paths = get_array(polydata, "Path")

        # Assuming `branch_ids` and `paths` are numpy arrays or lists.
        nonzero_indices = [i for i, path in enumerate(paths) if path != 0]
        nonzero_branch_ids = [branch_ids[i] for i in nonzero_indices]
        nonzero_paths = [paths[i] for i in nonzero_indices]

        for i in nonzero_indices:
            print(f"Branch ID: {branch_ids[i]}, Path: {paths[i]}")  

        branchingData = pd.read_csv(path + "/1D Input Files/branchingData.csv")
        child1 = branchingData["Child1"].to_numpy()
        child2 = branchingData["Child2"].to_numpy()
        parent = branchingData["Parent"].to_numpy()

        from collections import defaultdict
        from pprint import pprint

        tree = defaultdict(list)

        # Each row corresponds to a node
        for node in range(len(parent)):
            # Add children of this node (from child1 and child2)
            for child in (child1[node], child2[node]):
                if child != -1.0:  # filter out invalid children
                    tree[node].append(int(child))

        # Convert to regular dict for clean printing
        tree = dict(tree)

        # Assume `tree` is your dictionary with np.float64 keys and values: for printing only
        cleaned_tree = {}

        for k, vlist in tree.items():
            key = float(k)  # or int(k) if appropriate
            # Convert children to float, and filter out -1.0
            clean_children = [float(c) for c in vlist if c != -1.0]
            if clean_children:
                cleaned_tree[key] = clean_children

        # Pretty print the cleaned version
        pprint(cleaned_tree)


        centerlineVelocity = self.centerlineVel
        centerlineFlow = self.centerlineFlow
        print("Centerline velocity: ", centerlineVelocity[-1])
        # Average every 100 numbers
        chunk_size = 101 # For 100 points per segment- will later automate this
        self.averageVel = [] # stores the average steady state velocity for each segment
        self.averageFlow = [] # stores the average steady state flowrate for each segment

        for i in range(0, len(centerlineVelocity[0]), chunk_size):
            chunkVel = centerlineVelocity[-1][i:i+chunk_size]  # Take a slice of 10 numbers
            avgVel = sum(chunkVel) / len(chunkVel)  # Calculate the average
            self.averageVel.append(avgVel)
            print("Average velocity: ", avgVel)

            chunkFlow = centerlineFlow[-1][i:i+chunk_size]  # Take a slice of 10 numbers
            avgFlow = sum(chunkFlow) / len(chunkFlow)  # Calculate the average
            self.averageFlow.append(avgFlow)


        # Convert to Python floats and filter out -1.0: For printing only
        clean_arr = [float(x) for x in self.averageVel]

        # Optionally convert to ints
        # clean_arr = [int(x) for x in arr if x != -1.0]

        print(clean_arr)

        concentration = []

        from collections import deque
        self.c_val = np.full(100, 1.0)  # Initial concentration value for the root node
        def bfs(tree, start):
            queue = deque([(start, self.c_val)])  # Queue holds tuples: (node, c_val)
            
            while queue:
                node, c_val = queue.popleft()
                print("Visiting Segment: ", node)
                
                from advecSolver_copy import TransportSolver
                print("Path: ", nonzero_paths[int(node)])
                print("Average velocity: ", self.averageVel[int(node)])
                # print("c_val: ", len(c_val)) 
                # Create a transport solver with the c_val from the parent
                transport_sim = TransportSolver(
                    L=nonzero_paths[int(node)],
                    c_val=c_val,
                    u_val=self.averageVel[int(node)],
                    element_degree=k,
                    write_output=True
                )
                
                transport_sim.setup()
                snapshots, time = transport_sim.run()
                # print("Snapshots: ", snapshots)
                print(len([c_val[-1] for c_val in snapshots]))
                concentration.append(snapshots)

                # For each child, enqueue with this node's snapshots as c_val
                for child in tree.get(node, []):
                    queue.append((child, [c_val[-1] for c_val in snapshots])) # store only the concentration at the outlet
            print("Concentration values: ", len(concentration))
 
        bfs(tree, 0.0)

        # === Save labeled VTP at each time step ===
        output_dir = self.inputs["output_directory"]
        vtp_files = sorted(glob.glob(os.path.join(output_dir, "averaged*")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
        all_data = []
        i = 0

        def bf_append(tree, start):
            queue = deque([start])  # Queue holds tuples: (node, c_val)
            
            while queue:
                node, c_val = queue.popleft()
                print("Visiting Segment: ", node)
                
                for child in tree.get(node, []):
                    queue.append((child)) # store only the concentration at the outlet
            print("Concentration values: ", len(concentration))
        # for file in vtp_files:
        #     reader = vtk.vtkXMLPolyDataReader()
        #     reader.SetFileName(file)
        #     reader.Update()

        #     # Step 2: Access the existing polydata
        #     poly_data = reader.GetOutput()

        #     # Step 6: Append new data array to existing polydata
        #     append_data = vtk.vtkFloatArray()
        #     append_data.SetName("MyData")

        #     # Assuming you have a list of float values to add
        #     my_values = [row[i] for row in concentration]

        #     for val in my_values:
        #         append_data.InsertNextValue(val)

        #     poly_data.GetPointData().AddArray(append_data)

        #     # Step 7: Overwrite the same VTP file
        #     writer = vtk.vtkXMLPolyDataWriter()
        #     writer.SetFileName(file)  # Overwrite the same file
        #     writer.SetInputData(poly_data)
        #     writer.Write()
        #     i += 1
       
    def create_directory(self): # create directory for output files
        """Creates a directory if it doesn't exist."""
        dir = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/Output"
        psth = self.inputs["input_directory"]
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
        

################################## MAIN ##################################

if __name__ == "__main__":
    path="/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/Input/041725/Run3_1branches"
    setup= simulationInputs(input_directory=path)
    setup.create_directory()
    setup.load_vtp()
    # ref_model = pv.read(path + "/1D Input Files/1d_model.vtp")
    setup.centerlineVelocity()
    # output_path = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/Output/042425/Run4_042425"
    setup.detect_bifurcation()
    setup.create_sim_inputs()


