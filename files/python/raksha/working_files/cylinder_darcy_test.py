import gmsh
import numpy as np
import meshio
from mpi4py import MPI
from dolfinx.io import gmshio

gmsh.initialize()
gmsh.model.add("3D_cut_with_cylinder")

# Geometry parameters
L, H, W = 5.0, 3.0, 3.0
l = L
c_x, c_y, c_z = 0.0, H / 2.0, W / 2.0
r = 0.25
gdim = 3

mesh_comm = MPI.COMM_WORLD
model_rank = 0

# Create volume and subtract cylinder
if mesh_comm.rank == model_rank:
    cuboid = gmsh.model.occ.add_box(0, 0, 0, L, H, W)
    cylinder = gmsh.model.occ.add_cylinder(c_x, c_y, c_z, l, 0, 0, r)
    fluid, _ = gmsh.model.occ.cut([(3, cuboid)], [(3, cylinder)])
    gmsh.model.occ.synchronize()

    # Tag volume
    # fluid_marker = 1
    # gmsh.model.addPhysicalGroup(3, [fluid[0][1]], fluid_marker)
    # gmsh.model.setPhysicalName(3, fluid_marker, "Fluid")

    # Tag boundaries
    walls = []
    cylinder_wall = []
    boundaries = gmsh.model.getBoundary(fluid, oriented=False)

    for bdim, tag in boundaries:
        com = gmsh.model.occ.getCenterOfMass(bdim, tag)
        if np.allclose(com, [0, H/2, W/2]) or np.allclose(com, [L, H/2, W/2]) \
           or np.allclose(com, [L/2, 0, W/2]) or np.allclose(com, [L/2, H, W/2]) \
           or np.allclose(com, [L/2, H/2, 0]) or np.allclose(com, [L/2, H/2, W]):
            walls.append(tag)
        elif np.allclose(com, [L/2, H/2, W/2]):
            cylinder_wall.append(tag)

    wall_marker = 2
    cylinder_marker = 3
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(2, cylinder_wall, cylinder_marker)
    gmsh.model.setPhysicalName(2, cylinder_marker, "Cylinder_Wall")

    # Define distance-based mesh refinement near cylinder wall
    res_min = r / 3
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "SurfacesList", cylinder_wall)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # Mesh generation
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write("mesh.msh")
    msh = meshio.read("mesh.msh")
    meshio.write("mesh.xdmf", msh)

# Import into DOLFINx
domain, _, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
facet_tags.name = "Facet markers"

gmsh.finalize()
