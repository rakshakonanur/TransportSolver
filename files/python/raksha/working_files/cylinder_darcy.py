import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
import meshio
import numpy

gmsh.initialize()

L = 5.0
H = 3.0
W = 3.0
# l = 3.0
# c_x = L / 2.0 - l/2
l = L
c_x = 0.0
c_y = H / 2.0
c_z = W / 2.0
r = 0.25
gdim = 3



mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    cuboid = gmsh.model.occ.add_box(0, 0, 0, L, H, W, tag=1)
    cylinder = gmsh.model.occ.add_cylinder(c_x, c_y, c_z, l, 0, 0, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, cuboid)], [(gdim, cylinder)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    # print("Volumes: ", volumes)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(3, fluid_marker, "Fluid")
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
    # gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(gdim)
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    # gmsh.write("mesh.msh")
    # mesh = meshio.read("mesh.msh")
    # meshio.write("mesh.xdmf", mesh)
    # gmsh.fltk.run()          # Open Gmsh GUI to visualize

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem

fluid_marker, wall_marker, cylinder_marker = 1, 2, 3
fluid, walls, cylinder = [], [], []

if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    print("Boundaries: ", boundaries)
    # boundaries = gmsh.model.getBoundary(fluid[0], oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        print("Boundary: ", boundary[0], boundary[1], "Center of mass: ", center_of_mass)

        if np.allclose(center_of_mass, [0, H / 2, W / 2]) or np.allclose(center_of_mass, [L, H / 2, W / 2]) \
            or np.allclose(center_of_mass, [L / 2, H, W / 2]) or np.allclose(center_of_mass, [L / 2, 0, W / 2]) \
                or np.allclose(center_of_mass, [L / 2, H / 2, 0]) or np.allclose(center_of_mass, [L / 2, H / 2, W]):
            walls.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H/2, W/2]):
            cylinder.append(boundary[1])
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(2, cylinder, cylinder_marker)
    gmsh.model.setPhysicalName(2, cylinder_marker, "Cylinder_Wall") # refers to the walls of the cylinder
    # gmsh.model.addPhysicalGroup(1, fluid, fluid_marker)
    # gmsh.model.setPhysicalName(1, fluid_marker, "Fluid")

# Create distance field from cylinder.
# Add threshold of mesh sizes based on the distance field
# LcMax -                  /--------
#                      /
# LcMin -o---------/
#        |         |       |
#       Point    DistMin DistMax
res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "SurfacesList", cylinder)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 1 = Delaunay, 2 = Frontal, 3 = Frontal-Delaunay, 4 = Del2D, 5 = Del3D
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write("mesh.msh")
    mesh = meshio.read("mesh.msh")
    meshio.write("mesh.xdmf", mesh)

domain, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"


# k = 1
# Q_el = element("BDMCF", domain.basix_cell(), k)
# P_el = element("DG", domain.basix_cell(), k - 1)
# V_el = mixed_element([Q_el, P_el])
# V = functionspace(domain, V_el)

# (sigma, u) = TrialFunctions(V)
# (tau, v) = TestFunctions(V)

# x = SpatialCoordinate(domain)
# g = 10.0 * exp(-((x[0] - L/2) * (x[0] - L/2) + (x[1] - H/2) * (x[1] - H/2)) / 0.02) # Flux at inlet
# f = Constant(domain, PETSc.ScalarType(0.0), (gdim,)) # Source term

# dx = Measure("dx", domain)
# a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
# L = -inner(f, v) * dx

# class InletVelocity():
#     def __init__(self, t):
#         self.t = t

#     def __call__(self, x):
#         values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
#         values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
#         return values

