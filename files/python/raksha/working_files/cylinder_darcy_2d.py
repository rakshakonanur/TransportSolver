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
c_x = L / 2.0 
c_y = H / 2.0
c_z = W / 2.0
r = 0.05
gdim = 2

mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

wall_marker, obstacle_marker = 2, 3
walls, obstacle = [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        # print(center_of_mass)
        if np.allclose(center_of_mass, [0, H / 2, 0]) or np.allclose(center_of_mass, [L, H / 2, 0])\
             or np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
            
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
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
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

domain, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"
hf = ufl.CellDiameter(domain) # Cell diameter
alpha = dfx.fem.Constant(domain, dfx.default_scalar_type(100.0)) # SIPG penalty parameter
beta  = dfx.fem.Constant(domain, dfx.default_scalar_type(10.0)) # Nitsche penalty parameter
ds = ufl.Measure('ds', domain, subdomain_data=ft) # Exterior facet integrals

k = 1
Q_el = element("BDMCF", domain.basix_cell(), k)
P_el = element("DG", domain.basix_cell(), k - 1)
V_el = mixed_element([Q_el, P_el])
V = fem.functionspace(domain, V_el)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(domain)
f = 10.0 * exp(-((x[0] - L/2) * (x[0] - L/2) + (x[1] - H/2) * (x[1] - H/2)) / 0.05)

dx = Measure("dx", domain)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx

fdim = domain.topology.dim - 1


# facets_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], H))
# Q, _ = V.sub(0).collapse()
# dofs_top = fem.locate_dofs_topological((V.sub(0), Q), fdim, facets_top)


# def f1(x):
#     values = np.zeros((2, x.shape[1]))
#     values[1, :] = np.sin(0 * x[0])
#     return values


# f_h1 = fem.Function(Q)
# f_h1.interpolate(f1)
# bc_top = fem.dirichletbc(f_h1, dofs_top, V.sub(0))


# facets_bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
# dofs_bottom = fem.locate_dofs_topological((V.sub(0), Q), fdim, facets_bottom)


# def f2(x):
#     values = np.zeros((2, x.shape[1]))
#     values[1, :] = -np.sin(0 * x[0])
#     return values


# f_h2 = fem.Function(Q)
# f_h2.interpolate(f2)
# bc_bottom = fem.dirichletbc(f_h2, dofs_bottom, V.sub(0)
# bcs = [bc_top, bc_bottom])

P, _ = V.sub(1).collapse()
# dofs_cylinder = fem.locate_dofs_topological(P, fdim, ft.find(obstacle_marker))
# p_bc_fun = Function(P)
# p_bc_fun.x.array[:] = 100.0  # Initialize to 0

# bc_p_cylinder = dirichletbc(p_bc_fun, dofs_cylinder)

Q, _ = V.sub(0).collapse()
dofs_cylinder = fem.locate_dofs_topological(Q, fdim, ft.find(obstacle_marker))
v_bc_fun = Function(Q)
v_bc_fun.x.array[:] = 0.0  # Initialize to 0

bc_v_cylinder = dirichletbc(v_bc_fun, dofs_cylinder)

bcs = [bc_v_cylinder]
# bc_func = dfx.fem.Function(Q)
# # bc_func.interpolate(lambda x: 0.0 * x[0] + 0.0 * x[1])
# bc_func.x.array[:] = 0.0  # Initialize to 0
# # bc_func.y.array[:] = 0.0  # Initialize to 0
# L += beta / hf * bc_func* v * ds(obstacle_marker) # Weakly enforce Dirichlet BC using Nitsche's method
# a += beta / hf * sigma * v * ds(obstacle_marker)
# bcs = []

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
dim = domain.geometry.dim

# 1. Create a scalar Lagrange space for magnitude
# P1_el = element("Lagrange", domain.basix_cell(), degree=1)
# V_mag = fem.functionspace(domain, P1_el)
# sigma_out = Function(V_mag)
# sigma_out.interpolate(sigma_h)

with io.XDMFFile(domain.comm, "out_mixed_poisson/u.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u_h)


# from ufl import sqrt, inner  # Import required UFL operations

# # Create a scalar Lagrange function space for the magnitude
P1_el = element("Lagrange", domain.basix_cell(), degree=1)
# V_mag = fem.functionspace(domain, P1_el)
sigma_mag = dfx.fem.Function(dfx.fem.functionspace(domain, P1_el))
sigma_mag.interpolate(sigma_h)

# # Compute the magnitude of sigma_h as a UFL expression
# sigma_h_magnitude = sqrt(inner(sigma_h, sigma_h))


# # # Project the magnitude into the scalar Lagrange space
# # sigma_h_magnitude = fem.Function(V_mag)
# # projector = fem.petsc.LinearProblem(
# #     form(inner(TestFunctions(V_mag), sigma_h_magnitude_expr) * dx),
# #     sigma_h_magnitude,
# #     petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# # )
# # projector.solve()

# # Save the magnitude of sigma_h
with io.XDMFFile(domain.comm, "out_mixed_poisson/sigma_h_magnitude.xdmf", "w") as file:
    file.write_mesh(domain)  # Write the mesh
    file.write_function(sigma_mag)  # Write the magnitude of sigma_h