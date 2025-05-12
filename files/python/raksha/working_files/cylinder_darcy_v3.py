import gmsh
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook
import ufl
import dolfinx as dfx

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc, LinearProblem)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio, XDMFFile)
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
    fluid = gmsh.model.occ.cut([(gdim, cuboid)], [(gdim, cylinder)])
    gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(gdim)
    # gmsh.write("cylinder.msh")

    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    fluid_marker = 1

    walls = []
    cylinder_wall = []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)

    wall_marker, cylinder_marker = 2, 3
    walls, cylinder = [], []

    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    print("Boundaries: ", boundaries)
    # boundaries = gmsh.model.getBoundary(fluid[0], oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        print("Boundary: ", boundary[0], boundary[1], "Center of mass: ", center_of_mass)
        tol = 1e-4  # or even smaller if needed
        if np.allclose(center_of_mass, [0, H / 2, W / 2], atol=tol) or np.allclose(center_of_mass, [L, H / 2, W / 2], atol=tol) \
            or np.allclose(center_of_mass, [L / 2, H, W / 2], atol=tol) or np.allclose(center_of_mass, [L / 2, 0, W / 2], atol=tol) \
                or np.allclose(center_of_mass, [L / 2, H / 2, 0], atol=tol) or np.allclose(center_of_mass, [L / 2, H / 2, W], atol=tol):
            walls.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H/2, W/2]):
            cylinder.append(boundary[1])

    # gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    # gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    # gmsh.model.addPhysicalGroup(2, cylinder, cylinder_marker)
    # gmsh.model.setPhysicalName(2, cylinder_marker, "Cylinder_Wall")

    res_min = r / 3

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

    gmsh.finalize()

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

