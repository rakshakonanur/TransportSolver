import gmsh
import meshio
import pandas as pd
import numpy as np
from dolfinx.io import XDMFFile, gmshio
import os, subprocess, shutil
from mpi4py import MPI
import dolfinx as dfx
import vtk
import branch
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import glob
import re
from vtk.util.numpy_support import vtk_to_numpy
from basix.ufl import element
import pyvista as pv

WALL = 0
OUTLET = 1

def _get_ftetwild_path():
    path = os.environ.get("FTETWILD_PATH") or shutil.which("FloatTetwild_bin")
    if path is None:
        raise RuntimeError(
            "ERROR: cannot find fTetWild . "
            "Set $FTETWILD_PATH or add FloatTetwild_bin to your PATH."
        )
    return path

def geo_to_mesh_gmsh(geo_file, msh_file, mesh_size=0.001):
    gmsh.initialize()
    gmsh.open(geo_file)

    # Generate 1D mesh (use .generate(2) for 2D, .generate(3) for 3D)
    gmsh.model.mesh.generate(1)

    # Save mesh in MSH version 2 format
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh_file)

    gmsh.finalize()    

def stl_to_mesh_gmsh(stl_file, msh_file,char_len_min=0.001, char_len_max=0.025): # let final char_len_max be 0.01

    gmsh.initialize()
    gmsh.model.add("bioreactor")
    gmsh.merge(stl_file)

    gmsh.model.mesh.removeDuplicateNodes()
    
    # Classify surfaces to reconstruct the volume
    angle = 30  # angle threshold in degrees for feature edges
    force_param = True
    include_boundary = True
    curve_angle = 180
    gmsh.model.mesh.classifySurfaces(angle * (3.14159265359 / 180), include_boundary, force_param, curve_angle * (3.14159265359 / 180))
    
    # Create geometry from mesh
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.createTopology()

    # Create volume from surface
    s = gmsh.model.getEntities(2)  # Get all surfaces
    gmsh.model.geo.addSurfaceLoop([surf[1] for surf in s], 1)
    vol = gmsh.model.geo.addVolume([1], 1)
    gmsh.model.geo.synchronize()

    # Define physical group (optional but needed for DOLFINx)
    gmsh.model.addPhysicalGroup(3, [1], 1)  # Volume
    gmsh.model.setPhysicalName(3, 1, "volume")

    p = gmsh.model.addPhysicalGroup(3, [vol], 9)
    gmsh.model.setPhysicalName(3, p, "Wall")

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len_min)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_file)
    logger.info(f"Created mesh {msh_file} from {stl_file}")
    gmsh.finalize()

def stl_to_mesh_ftet(stl_file, msh_file, edge_length=0.007, eps=0.0005):
    edge_length = 0.007  # ideal_edge_length = diag_of_bbox * L. (double, optional, default: 0.05) GOOD 0.007
    eps = 0.0005 # epsilon = diag_of_bbox * EPS. (double, optional, default: 1e-3) GOOD 0.0005
    float_tetwild_path = _get_ftetwild_path()
    command = f"{float_tetwild_path} -i {stl_file} -o {msh_file} --lr {edge_length} --epsr {eps}"
    subprocess.run(command, shell=True, check=True)

def mesh_to_xdmf(msh_file, xdmf_file):
    # Convert .msh to .vtu
    mesh = meshio.read(msh_file)
    meshio.write(xdmf_file, mesh)

def xdmf_to_dolfinx(xdmf_file):
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

def convert_mesh(msh_file, xdmf_file):
    """
    Read the mesh from a file and convert it to XDMF format.
    """
    # Read full Gmsh mesh
    msh = meshio.read(msh_file)
    # Extract only "line" elements
    line_cells = [cell for cell in msh.cells if cell.type == "line"]
    if not line_cells:
        raise RuntimeError("No line cells found in the mesh.")
    # meshio expects a list of (cell_type, numpy_array)
    line_cells = [("line", line_cells[0].data)]
    # Optionally, also filter cell data for lines only
    line_cell_data = {}
    if "gmsh:physical" in msh.cell_data_dict:
        # meshio expects a dict of {cell_type: [data_array]}
        line_cell_data = {"gmsh:physical": [msh.cell_data_dict["gmsh:physical"]["line"]]}
    # Create a new mesh with only line elements
    line_mesh = meshio.Mesh(
        points=msh.points,
        cells=line_cells,
        cell_data=line_cell_data if line_cell_data else None
    )
    # Write to XDMF
    line_mesh.write(xdmf_file)

def branch_mesh_tagging(mesh):
    
    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    boundary_facets_indices = dfx.mesh.exterior_facet_indices(mesh.topology)
    inlet = np.array([0,0.41,0.34]) #updated with units
    tol = 1e-6

    def near_inlet(x):
        return np.isclose(x[0], inlet[0]) & np.isclose(x[1], inlet[1]) & np.isclose(x[2], inlet[2])

    inlet_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, near_inlet)

    # Extract outlet facets: boundary facets excluding inlet facets
    outlet_facets = np.setdiff1d(boundary_facets_indices, inlet_facets)

    facet_indices = np.concatenate([inlet_facets, outlet_facets])
    facet_markers = np.concatenate([np.full(len(inlet_facets), 1, dtype=np.int32),
                                    np.full(len(outlet_facets), 2, dtype=np.int32)])
    facet_tag = dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_markers)
    print("Facet indices: ", facet_indices, flush=True)
    print("Facet markers: ", facet_markers, flush=True)

    # Return outlet coordinates
    outlet_coords = mesh.geometry.x[facet_tag.indices[facet_tag.values == 2]]
    print("Outlet coordinates: ", outlet_coords, flush=True)

    return outlet_coords, facet_tag

def generate_1d_files(xdmf_file, output_dir):

    # Load the converted XDMF mesh
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    P1 = dfx.fem.functionspace(mesh, ("CG", 1))  # Continuous Lagrange, degree 1
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)  # Vector space for velocity
    dof_coords = P1.tabulate_dof_coordinates()
    num_dofs = dof_coords.shape[0]

    velocity_fn = dfx.fem.Function(P1vec)
    pressure_fn = dfx.fem.Function(P1)

    centerlineVel, centerlineFlow, pressure, centerlineCoords = load_vtp(output_dir)
    Nt = len(centerlineVel)  # Number of timesteps

    xdmf_vel = dfx.io.XDMFFile(mesh.comm, "velocity.xdmf", "w")
    xdmf_pressure = dfx.io.XDMFFile(mesh.comm, "pressure.xdmf", "w")

    xdmf_vel.write_mesh(mesh)
    xdmf_pressure.write_mesh(mesh)

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    outlet_coords, facet_tag = branch_mesh_tagging(mesh)
    xdmf_vel.write_meshtags(facet_tag, mesh.geometry)
    xdmf_pressure.write_meshtags(facet_tag, mesh.geometry)

    # Allocate u_val as a 2D array for all timesteps and dofs
    u_val = np.zeros((Nt, num_dofs))

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim,
                                       mesh.topology.dim - 1)

    # Commenting out nearest neighbor search
    tree = None  # KDTree for nearest neighbor search
    dt = 1
    
    for i in range(Nt):
        time = i * dt
        points = centerlineCoords  # Get the points for the current timestep
        scalar_velocity = centerlineVel[time]  # Get the velocity for the current timestep
        scalar_pressure = pressure[time]  # Get the pressure for the current timestep

        if tree is None:
            tree = cKDTree(points)

        # Nearest neighbor interpolation
        distances, indices = tree.query(dof_coords)
        interpolated_velocity = scalar_velocity[indices]
        interpolated_pressure = scalar_pressure[indices]

        # Compute tangents using the next connected point
        tangents = np.zeros((len(indices), 3))
        for j, idx in enumerate(indices):
            # Simple forward difference: pick next point if possible, else previous
            if idx < len(points) - 1:
                delta = points[idx + 1] - points[idx]
            else:
                delta = points[idx] - points[idx - 1]
            tangent = delta / np.linalg.norm(delta)
            tangents[j] = tangent

        # Assign to velocity function
        velocity_fn.x.array[:] = (interpolated_velocity[:, np.newaxis] * tangents).flatten()
        pressure_fn.x.array[:] = interpolated_pressure

        # --- Write data to file with time stamp ---

        xdmf_vel.write_function(velocity_fn, time)
        xdmf_pressure.write_function(pressure_fn, time)

    return outlet_coords

    
def load_vtp(directory):
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
    points_per_section = 20  # Number of points per section to average

    centerlineVel = []  # initialize as a list
    centerlineFlow = []  # initialize as a list
    pressure = []  # initialize as a list

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


        area = timestep_data["point_data"]["Area"]  # (N,)
        flowrate_1d = timestep_data["point_data"]["Flowrate"]  # (N,)
        reynolds_1d = timestep_data["point_data"]["Reynolds"]  # (N,)
        pressure_1d = timestep_data["point_data"]["Pressure_mmHg"]  # (N,)
        mesh_3d_coords = timestep_data["coords"]  # (N, 3)

        velocity_1d = flowrate_1d/area # calculate velocity from flowrate and area
        centerline = velocity_1d[::points_per_section] # save only one point for each cross-section
        centerlineFlowrate = flowrate_1d[::points_per_section] # save only one point for each cross-section
        centerlineFlow.append(centerlineFlowrate) # save the flowrate values for each timestep in each row

        centerlineVel.append(centerline)# save the velocity values for each timestep in each row
        pressure.append(pressure_1d[::points_per_section]) # save the pressure values for each timestep in each row
        centerlineCoords = mesh_3d_coords.reshape(-1, points_per_section, 3).mean(axis=1)


    return centerlineVel, centerlineFlow, pressure, centerlineCoords


def domain_mesh_tagging(mesh, coords, tag_value=1, tol=5e-3):

    """
    Tag vertices in the mesh that are near given coordinates.

    Parameters:
        mesh (dolfinx.Mesh): The mesh object
        coords (np.ndarray): An (N, 3) array of points to tag vertices near
        tag_value (int): The marker value to assign to those vertices
        tol (float): Tolerance for proximity check

    Returns:
        dolfinx.mesh.meshtags: Vertex tags
    """
    vdim = 0  # vertex dimension
    mesh_coords = mesh.geometry.x

    # Define proximity function
    def near_coords(x):
        return np.any([np.linalg.norm(x.T - p, axis=1) < tol for p in coords], axis=0)

    # Find matching vertex indices
    near_vertex_indices = dfx.mesh.locate_entities(mesh, vdim, near_coords)

    # Tag them
    tags = np.full_like(near_vertex_indices, tag_value, dtype=np.int32)
    vertex_tags = dfx.mesh.meshtags(mesh, vdim, near_vertex_indices, tags)

    print("Tagged vertex indices:", near_vertex_indices, flush=True)
    # Ensure vertex-to-cell connectivity exists
    mesh.topology.create_connectivity(0, mesh.topology.dim)

    # Write vertex tags
    with dfx.io.XDMFFile(mesh.comm, "vertex_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(vertex_tags, mesh.geometry)

    return vertex_tags

from scipy.spatial import cKDTree

def domain_mesh_tagging_nearest(xdmf_file, coords):
    """
    Tag the nearest vertex in the mesh to each coordinate in coords.

    Parameters:
        mesh (dolfinx.mesh.Mesh): The mesh object
        coords (np.ndarray): An (N, 3) array of target points
        tag_value (int): The marker value to assign

    Returns:
        dolfinx.mesh.meshtags: Vertex tags
    """

    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    
    mesh.topology.create_connectivity(0, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    vdim = 0  # Vertex dimension
    fdim = mesh.topology.dim - 1  # Facet dimension
    mesh_coords = mesh.geometry.x

    # Build KDTree from mesh coordinates
    tree = cKDTree(mesh_coords)
    distances, vertex_indices = tree.query(coords)

    # Remove duplicates (optional)
    unique_vertex_indices = np.unique(vertex_indices)

    # Create tags
    tags = np.full_like(unique_vertex_indices, OUTLET, dtype=np.int32)
    vertex_tags = dfx.mesh.meshtags(mesh, vdim, unique_vertex_indices, tags)

    print("Nearest tagged vertex indices:", unique_vertex_indices, flush=True)
    print("Vertex coordinates:", mesh_coords[unique_vertex_indices], flush=True)

    # Ensure vertex-to-cell connectivity for writing
    mesh.topology.create_connectivity(0, mesh.topology.dim)

    # Write to file
    with dfx.io.XDMFFile(mesh.comm, xdmf_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(vertex_tags, mesh.geometry)

        print(f"Total facets: {mesh.topology.index_map(fdim).size_local}")
        print(f"Tagged facets: {vertex_tags.indices.size}")


    return vertex_tags

def import_branched_mesh(branching_data_file, geo_file="branched_network.geo", msh_file="branched_network.msh", xdmf_file="branched_network.xdmf"):
    df = pd.read_csv(branching_data_file)
    branch.write_geo_from_branching_data(df, geo_file=geo_file)
    geo_to_mesh_gmsh(geo_file=geo_file, msh_file=msh_file)
    convert_mesh(msh_file=msh_file, xdmf_file=xdmf_file)
    xdmf_to_dolfinx(xdmf_file=xdmf_file)
    outlet_coords = generate_1d_files(xdmf_file=xdmf_file, output_dir="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output/1D_Output/071725/Run5_25branches")
    return outlet_coords

def create_bioreactor_mesh(stl_file, msh_file="bioreactor.msh", xdmf_file="bioreactor.xdmf", diric=None):
    stl_to_mesh_gmsh(stl_file, msh_file=msh_file)
    mesh_to_xdmf(msh_file=msh_file, xdmf_file=xdmf_file)
    xdmf_to_dolfinx(xdmf_file=xdmf_file)
    facet_tags = domain_mesh_tagging_nearest(xdmf_file=xdmf_file, coords=diric)


class Files():
    def __init__(self, stl_file, branching_data_file):
        dirichlet = import_branched_mesh(branching_data_file)
        create_bioreactor_mesh(stl_file, diric=dirichlet)
        self.D_value = 1e-2  # Diffusion coefficient
        self.element_degree = 1  # Polynomial degree for finite elements
        self.write_output = True  # Whether to write output files   



if __name__ == "__main__":
    perfusion = Files(stl_file="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/syntheticVasculature/files/geometry/cermRaksha_scaled.stl",
                                branching_data_file="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output/1D_Output/071725/Run5_25branches/1D_Input_Files/branchingData.csv")


