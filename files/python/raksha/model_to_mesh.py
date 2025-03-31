import vtk
import os
import argparse
import meshio
import subprocess


def vtp_to_stl(vtp_file, output_directory):
    # Read the VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    poly_data = reader.GetOutput()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Write to STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_directory+"/model.stl")
    writer.SetInputData(poly_data)
    writer.Write()


def stl_to_msh(stl_file, output_directory):
    edge_length = 0.007  # ideal_edge_length = diag_of_bbox * L. (double, optional, default: 0.05) GOOD 0.007
    eps = 0.0005 # epsilon = diag_of_bbox * EPS. (double, optional, default: 1e-3) GOOD 0.0005
    float_tetwild_path = "/mnt/c/Users/rkona/Documents/advectionDiffusion/fTetWild/build/FloatTetwild_bin"
    command = f"{float_tetwild_path} -i {stl_file} -o {output_directory} --lr {edge_length} --epsr {eps}"
    subprocess.run(command, shell=True, check=True)


def msh_to_xdmf(msh_file, output_directory):

    # Convert .msh to .vtu
    mesh = meshio.read(msh_file)
    meshio.write(output_directory + "_before.vtu", mesh)


    # Step 2: Apply vtkConnectivityFilter to keep the largest connected component
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(output_directory + "_before.vtu")
    reader.Update()
    input_mesh = reader.GetOutput()

    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(input_mesh)
    connectivity_filter.SetExtractionModeToLargestRegion()  # Keep the largest component
    connectivity_filter.Update()

    # Step 3: Write the filtered mesh to a new .vtu file

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_directory + ".vtu")
    writer.SetInputData(connectivity_filter.GetOutput())
    writer.Write()


    # Step 4: Convert the filtered .vtu to .xdmf using meshio
    filtered_mesh = meshio.read(output_directory + ".vtu")
    meshio.write(output_directory + ".xdmf", filtered_mesh)


def main(model_directory, output_directory):

    vtp_to_stl(model_directory, output_directory)
    stl_to_msh(output_directory + "/model.stl", output_directory + ".msh")
    msh_to_xdmf(output_directory + ".msh", output_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process centerline tracing.")
    parser.add_argument("model_directory", type=str, help="Path to the model directory.")
    parser.add_argument("output_directory", type=str, help="Directory to save the results.")

    args = parser.parse_args()
    main(args.model_directory, args.output_directory)



# def msh_to_xdmf(msh_file, output_directory):

#     # Read the 3D .msh file
#     msh = meshio.read(msh_file)

#     # Separate the 3D volume elements and surface (boundary) elements
#     volume_mesh = meshio.Mesh(
#         points=msh.points,
#         cells={"tetra": msh.get_cells_type("tetra")}
#     )
#     boundary_mesh = meshio.Mesh(
#         points=msh.points,
#         cells={"triangle": msh.get_cells_type("triangle")}
#     )

#     # Preserve physical groups if they exist
#     if "gmsh:physical" in msh.cell_data_dict:
#         if "tetra" in msh.cell_data_dict["gmsh:physical"]:
#             volume_mesh.cell_data["gmsh:physical"] = [
#                 msh.cell_data_dict["gmsh:physical"]["tetra"]
#             ]
#         if "triangle" in msh.cell_data_dict["gmsh:physical"]:
#             boundary_mesh.cell_data["gmsh:physical"] = [
#                 msh.cell_data_dict["gmsh:physical"]["triangle"]
#             ]

#     # Export the XDMF mesh
#     def export_xdmf(mesh, file_path):
#         meshio.write(file_path, mesh)

#     # Export the volume mesh and boundary mesh to XDMF
#     export_xdmf(volume_mesh, output_directory + ".xdmf")


# def vtu_to_stl(vtu_file, output_directory):
#     # Read the VTP file
#     reader = vtk.vtkXMLUnstructuredGridReader()
#     reader.SetFileName(vtu_file)
#     reader.Update()
#     poly_data = reader.GetOutput()

#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     # Write to STL
#     writer = vtk.vtkSTLWriter()
#     writer.SetFileName(output_directory+"/model.stl")
#     writer.SetInputData(poly_data)
#     writer.Write()


# input_file = "/Users/galasanchezvanmoer/Desktop/PhD_Project/GitHub_repositories/Eikonal_mine/triangulated_mesh.vtu"
