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
    fluid = gmsh.model.occ.cut([(gdim, cuboid)], [(gdim, cylinder)])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    fluid_marker = 1
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(3, fluid_marker, "Fluid")

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
    gmsh.model.addPhysicalGroup(2, walls, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(2, cylinder, cylinder_marker)
    gmsh.model.setPhysicalName(2, cylinder_marker, "Cylinder_Wall")

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
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write("mesh.msh")
    mesh = meshio.read("mesh.msh")
    meshio.write("mesh.xdmf", mesh)

print(model_rank)
domain, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"
gmsh.finalize()

k = 2
# Q_el = element("BDM", domain.basix_cell(), k-1)
Q_el = element("BDM", domain.basix_cell(), k)

P_el = element("Lagrange", domain.basix_cell(), k)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(domain, V_el)
hf = ufl.CellDiameter(domain) # Cell diameter
beta  = Constant(domain, dfx.default_scalar_type(10.0)) # Nitsche penalty parameter

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(domain)
# f = Constant(domain, dfx.default_scalar_type(0.0)) 
f = 10.0 * exp(-((x[1] - H/2) * (x[1] - H/2) + (x[2] - W/2) * (x[2] - W/2)) / 0.2)

dx = Measure("dx", domain)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx  

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
# scalar = np.array((1,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
# bc_cylinder = dirichletbc(scalar, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)

# Q, _ = V.sub(0).collapse()
# dofs_cylinder = fem.locate_dofs_topological(Q, fdim, ft.find(cylinder_marker))
# v_bc_fun = Function(Q)
# v_bc_fun.x.array[:] = 0.0  # Initialize to 0
# # v_bc_fun.x.array[:] = 1.0  # You may want to selectively assign only boundary DOFs!

# # Apply to the dofs on the cylinder
# bc_v_cylinder = dirichletbc(v_bc_fun, dofs_cylinder)
# bc_v_cylinder = dirichletbc(value, dofs_cylinder, V.sub(0))

P, _ = V.sub(1).collapse()
# Pressure at cylinder wall (p = 1)
cylinder_facets = ft.find(cylinder_marker)
dofs_p_cyl = fem.locate_dofs_topological(P, fdim, cylinder_facets)
bc_p_cyl = fem.dirichletbc(PETSc.ScalarType(100.0), dofs_p_cyl, V.sub(1))

# Pressure at outer walls (p = 0)
wall_facets = ft.find(wall_marker)
dofs_p_wall = fem.locate_dofs_topological(P, fdim, wall_facets)
bc_p_wall = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_p_wall, V.sub(1))

bcs = [bc_p_cyl, bc_p_wall]

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                      "pc_factor_mat_solver_type": "mumps"})
try:
    w_h = problem.solve()
except PETSc.Error as e:  # type: ignore
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

sigma_h, u_h = w_h.split()
from basix.ufl import element
with io.XDMFFile(domain.comm, "out_mixed_poisson/u.xdmf", "w") as file:
    file.write_mesh(domain)
    u_out = dfx.fem.Function(dfx.fem.functionspace(domain, P_el))
    u_out.interpolate(u_h)
    file.write_function(u_out)
