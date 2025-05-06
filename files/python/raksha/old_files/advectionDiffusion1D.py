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
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET

def load_pvd(directory):
    
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
    
def load_vtp(directory):

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
    return all_data

def maximum_inscribed_radius(path_name, pointID):
    """
    Import a mesh and its corresponding facet tags from centerline VTP files.

    Args:
        path_mesh (str): Path to the VTP file containing the mesh.
        path_facets (str): Path to the VTP file containing the facet tags.

    Returns:
        dolfinx.cpp.mesh.Mesh: The imported mesh.
    """

    # Read the mesh from the VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path_name)
    reader.Update()
    polydata = reader.GetOutput()

    # Get the point data array names
    point_data = polydata.GetPointData()
    print("Available point data arrays:")
    for i in range(point_data.GetNumberOfArrays()):
        print(f"  - {point_data.GetArrayName(i)}")

    # Get the 'MaximumInscribedRadius' array
    radii_array = point_data.GetArray("MaximumInscribedSphereRadius")

    # Choose a point index (for example, point 0)
    point_index = pointID  # Replace with the desired point index
    radius_at_point = radii_array.GetValue(point_index)
    coords = polydata.GetPoint(point_index)
    print(f"Coordinates of point {point_index}: {coords}")
    print(f"Minimum inscribed radius at point {point_index}: {radius_at_point}")
    return coords, radius_at_point

from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI

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

def export_xdmf(path_export, mesh, function):
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

import numpy as np
import meshio
from scipy.interpolate import griddata

def convert_1d_to_3d(data_series):
    # Step 1: Load 3D mesh using meshio
    mesh = meshio.read(r"C:\Users\rkona\Documents\advectionDiffusionFiles\041725\Run3_1branches\Results\merged_model\merged_model.xdmf")
    coords_3d = mesh.points  # (M, 3)

    all_data = []
    print("Number of timesteps:", len(data_series))
    # Iterate over each timestep in the data series
    for i in range(len(data_series)):
        print(f"Processing timestep {i}")
        # Step 2: Extract 1D data (pressure) from the each timestep
        pressure_1d = data_series[i]["point_data"]["Pressure_mmHg"]  # (N,)
        flowrate_1d = data_series[i]["point_data"]["Flowrate"]  # (N,)
        disps_1d = data_series[i]["point_data"]["Disps"]  # (N,)
        area_1d = data_series[i]["point_data"]["Area"]  # (N,)
        reynolds_1d = data_series[i]["point_data"]["Reynolds"]  # (N,)
        wss_1d = data_series[i]["point_data"]["WSS"]  # (N,)
        coords_1d = data_series[i]["coords"]  # (N, 3)

        # Step 3: Interpolate 1D data to 3D mesh points
        # You can use method='linear' or 'nearest'
        pressure_3d = griddata(coords_1d, pressure_1d, coords_3d, method='linear')
        flowrate_3d = griddata(coords_1d, flowrate_1d, coords_3d, method='linear')
        disps_3d = griddata(coords_1d, disps_1d, coords_3d, method='linear')
        area_3d = griddata(coords_1d, area_1d, coords_3d, method='linear')
        reynolds_3d = griddata(coords_1d, reynolds_1d, coords_3d, method='linear')
        wss_3d = griddata(coords_1d, wss_1d, coords_3d, method='linear')

        # Optional fallback for NaNs from linear interpolation
        # Fill NaNs with nearest neighbor if needed
        nan_mask = np.isnan(pressure_3d)
        if np.any(nan_mask):
            pressure_3d[nan_mask] = griddata(coords_1d, pressure_1d, coords_3d[nan_mask], method='nearest')

        nan_mask = np.isnan(flowrate_3d)
        if np.any(nan_mask):
            flowrate_3d[nan_mask] = griddata(coords_1d, flowrate_1d, coords_3d[nan_mask], method='nearest')    
        
        # Calculate the magnitude of the displacement vector
        disps_magnitude = np.linalg.norm(disps_3d, axis=1)  # Compute the magnitude of each displacement vector

        # Create a mask for NaN values in the magnitude
        nan_mask = np.isnan(disps_magnitude)
        if np.any(nan_mask):
            disps_3d[nan_mask] = griddata(coords_1d, disps_1d, coords_3d[nan_mask], method='nearest')
        
        nan_mask = np.isnan(area_3d)
        if np.any(nan_mask):
            area_3d[nan_mask] = griddata(coords_1d, area_1d, coords_3d[nan_mask], method='nearest')

        nan_mask = np.isnan(reynolds_3d)
        if np.any(nan_mask):
            reynolds_3d[nan_mask] = griddata(coords_1d, reynolds_1d, coords_3d[nan_mask], method='nearest')

        nan_mask = np.isnan(wss_3d)
        if np.any(nan_mask):
            wss_3d[nan_mask] = griddata(coords_1d, wss_1d, coords_3d[nan_mask], method='nearest')

        timestep_data = {
                "coords": coords_3d,
                "pressure": pressure_3d,
                "flowrate": flowrate_3d,
                "disps": disps_3d,
                "area": area_3d,
                "reynolds": reynolds_3d,
                "wss": wss_3d
            }
        all_data.append(timestep_data)

    output_dir = "output_vtu_meshio_7"
    os.makedirs(output_dir, exist_ok=True)

    for i, timestep in enumerate(all_data):
        coords = timestep["coords"]  # shape (M, 3)
        pressure = timestep["pressure"]
        flowrate = timestep["flowrate"]
        disps = timestep["disps"]
        area = timestep["area"]
        reynolds = timestep["reynolds"]
        wss = timestep["wss"]

        mesh = meshio.Mesh(
            points=coords_3d,               # 3D mesh coordinates
            cells=mesh.cells,               # use original cells from .xdmf
            point_data={"Pressure_mmHg": pressure, "Flowrate": flowrate, "Disps": disps, "Area": area, "Reynolds": reynolds, "WSS": wss}  # 1D data mapped to 3D mesh
        )

        filename = os.path.join(output_dir, f"mapped3d_t{i:04d}.xdmf")
        meshio.write(filename, mesh)
        print(f"Saved {filename}")

def create_xdmf_master(path): # NOT WORKING
    """
    Create a master XDMF file that references all the individual timestep files.

    Args:
        path (str): Path to the directory containing the timestep files.
    Returns:
        None 
        
    """
    
    # === Config ===
    input_dir = path  # Path to folder with existing .xdmf files
    output_master = os.path.join(input_dir, "time_series_master.xdmf")

    # Automatically list and sort the XDMF files
    xdmf_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".xdmf") and f.startswith("mapped3d")
    ])
    print(f"Found {len(xdmf_files)} XDMF files.")
    # Create dummy time values (replace with actual ones if you have them)
    time_values = [i * 0.01 for i in range(len(xdmf_files))]

    # === Write master XDMF ===
    with open(output_master, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')

        for i, (filename, time_val) in enumerate(zip(xdmf_files, time_values)):
            print(f"Adding timestep {i}: {filename} with time value {time_val}")
            f.write(f'      <Grid Name="timestep_{i}">\n')
            f.write(f'        <xi:include href="{filename}" xpointer="xpointer(//Grid)"/>\n')
            f.write(f'        <Time Value="{time_val}"/>\n')
            f.write('      </Grid>\n')

        f.write('    </Grid>\n')
        f.write('  </Domain>\n')
        f.write('</Xdmf>\n')

    print(f"✅ Master XDMF created: {output_master}")

import dolfinx
    
def create_mesh_tags(path):
    """
    Create mesh tags for the XDMF file.

    Args:
        path (str): Path to the XDMF file.

    Returns:
        None
    """
    
    domain = import_mesh(path)

    # Create connectivity between the mesh elements and their facets
    domain.topology.create_connectivity(domain.topology.dim, 
                                        domain.topology.dim - 1)
    
    # Facet tags
    bc_facet_indices, bc_facet_markers = [], []

    # MESHTAGS
    INLET    = 1
    OUTLET   = 2
    WALL     = 0

    wall_tags(domain, bc_facet_indices, bc_facet_markers, WALL)
    path = "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches/1D Input Files/centerlines.vtp"
    coords, inlet_radius = maximum_inscribed_radius(path, 0)
    inlet_tags(domain, bc_facet_indices, bc_facet_markers, coords, inlet_radius, INLET)

def wall_tags(domain, bc_facet_indices, bc_facet_markers, WALL):
    """
    Create wall tags for the mesh.

    Args:
        domain: The mesh domain.

    Returns:
        None
    """
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)
    
    # Get boundary facets (2D triangles) and corresponding DOFs
    boundary_facets_indices = dolfinx.mesh.exterior_facet_indices(domain.
                                                                    topology)
    bc_facet_indices.append(boundary_facets_indices)
    bc_facet_markers.append(np.full_like(boundary_facets_indices, WALL))
    print("Wall tags created: ", len(boundary_facets_indices))

from scipy.spatial import cKDTree

def inlet_tags(domain, bc_facet_indices, bc_facet_markers, coords, radius, INLET):
    """
    Create inlet tags for the mesh.

    Args:
        domain: The mesh domain.
        coords: Coordinates of the centerline point.
        radius: Radius of the inlet from the MaximumInscribedSphereRadius.

    Returns:
        None
    """
    domain.topology.create_connectivity(domain.topology.dim - 1,
                                        domain.topology.dim)
    
    search_radius = 1.5 * radius
    
    coordinates = domain.geometry.x
    print("Coordinates of the mesh: ", coordinates.shape)
    # Build a KDTree for the mesh coordinates
    tree = cKDTree(coordinates)

    # Query points within radius
    indices = tree.query_ball_point(coords, r=search_radius)

    # Print results
    print(f"Found {len(indices)} mesh points within {search_radius} units of the target point.")

def fill_caps(vtp_filename, output_filename):
    
    # Read the .vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_filename)
    reader.Update()
    polydata = reader.GetOutput()

    # Find open edges (i.e., boundaries)
    boundaries = vtk.vtkFeatureEdges()
    boundaries.SetInputData(polydata)
    boundaries.BoundaryEdgesOn()
    boundaries.FeatureEdgesOff()
    boundaries.ManifoldEdgesOff()
    boundaries.NonManifoldEdgesOff()
    boundaries.Update()

    # Triangulate the open boundaries
    stripper = vtk.vtkStripper()
    stripper.SetInputData(boundaries.GetOutput())
    stripper.Update()

    boundary_poly = vtk.vtkPolyData()
    boundary_poly.SetPoints(stripper.GetOutput().GetPoints())
    boundary_poly.SetLines(stripper.GetOutput().GetLines())

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(boundary_poly)
    triangulator.Update()

    # Append caps to original mesh
    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(polydata)
    append_filter.AddInputData(triangulator.GetOutput())
    append_filter.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(append_filter.GetOutput())
    cleaner.Update()

    # Write the result
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(cleaner.GetOutput())
    writer.Write()

    print(f"✅ Saved capped VTP to: {output_filename}")

fill_caps("/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches/3d_tmp/merged_model.vtp",
          "/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches/Results_caps/merged_model_caps.vtp")
        
        



# data_series = load_pvd(r"C:\Users\rkona\Documents\advectionDiffusionFiles\041725\Run3_1branches")
# convert_1d_to_3d(data_series)

# Ubuntu

# data_series = load_vtp("/mnt/c/Users/rkona/Documents/advectionDiffusionFiles/041725/Run3_1branches")
# path = "/mnt/c/Users/rkona/Documents/advectionDiffusion/output_vtu_meshio_7"
# create_xdmf_master(path)

# Test mesh tags
# path = "/mnt/c/Users/rkona/Documents/advectionDiffusion/output_vtu_meshio_7/mapped3d_t0000.xdmf"
# create_mesh_tags(path)

# arrays = load_pvd(r"C:\Users\rkona\Documents\advectionDiffusionFiles\041025\Run1_1branches")
# # arrays = load_vtp(r"C:\Users\rkona\Documents\advectionDiffusionFiles\041025\Run1_1branches")

# # Example: How to load in flowrate data at t = 100 at all points
# print(arrays[100]["point_data"]["Flowrate"])
# print(arrays[100]["point_data"]["Flowrate"].shape)

# # # Example: How to load in point data at t = 100 (constant across all timesteps)
# # print(arrays[100]["coords"])

