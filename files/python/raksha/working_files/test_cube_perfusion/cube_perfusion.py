import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl               import avg, jump, dot, grad
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx import (fem, io, mesh)
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting, LinearProblem, set_bc
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from typing import List, Optional
from dolfinx.io import XDMFFile

print = PETSc.Sys.Print

# MESHTAGS
LEFT = 1
RIGHT = 2
TOP = 3
BOTTOM = 4
WALL = 5


# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def create_cube_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, nx = N_cells, ny = N_cells, nz = N_cells,
                                           cell_type = dfx.mesh.CellType.tetrahedron,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], 1.0)
        def wall(x):
            return np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOTTOM))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        wall_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, wall)
        bc_facet_indices.append(wall_BC_facets) # Wall facets
        bc_facet_markers.append(np.full_like(wall_BC_facets, WALL)) 

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags

class PerfusionSolver:

    def __init__(self, N_cells: int,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.N = N_cells
        self.mesh, self.ft = create_cube_mesh_with_tags(N_cells=N_cells)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output

    def setup(self):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        # k = self.element_degree
        k = 1
        P_el = element("Lagrange", self.mesh.basix_cell(), k)
        u_el = element("DG", self.mesh.basix_cell(), k-1, shape=(self.mesh.geometry.dim,))
        dx = Measure("dx", self.mesh)

        # Define function spaces
        W = dfx.fem.functionspace(self.mesh, P_el) # Pressure function space
        V = dfx.fem.functionspace(self.mesh, u_el) # Velocity function space

        kappa = 1
        mu = 1
        kappa_over_mu = fem.Constant(self.mesh, dfx.default_scalar_type(kappa/mu))
        phi = fem.Constant(self.mesh, dfx.default_scalar_type(0.1)) # Porosity of the medium, ranging from 0 to 1
        S = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term

        # Trial and test functions
        v, p = ufl.TestFunction(W), ufl.TrialFunction(W)

        # Boundary conditions
        bc_left  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1.0))
        bc_right = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        dof_left  = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0.0))
        dof_right = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 1.0))
        bcs = [dfx.fem.dirichletbc(bc_left, dof_left, W),
                dfx.fem.dirichletbc(bc_right, dof_right, W)
        ]

        # Define variational problem
        a = kappa_over_mu * dot(grad(p), grad(v)) * ufl.dx 
        f = S * v * ufl.dx  # residual form of our equation

        self.a = a
        self.f = f

        # Apply Dirichlet BCs
        self.bcs = bcs

        problem = LinearProblem(self.a, self.f, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

        try:
            # projector = Projector(f, V, bcs = [])
            p_h = problem.solve()
            # fig = plt.figure()
            # im = plot(vf)
            # plt.colorbar(im, format="%.2e")
        except PETSc.Error as e:  # type: ignore
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

        # sigma_h, u_h = w_h.split()

        projector = Projector(V)
        vel_f = projector(-kappa_over_mu * grad(p_h) / phi)

        with XDMFFile(self.mesh.comm, "out_darcy/p.xdmf","w") as file:
            file.write_mesh(self.mesh)
            file.write_function(p_h, 0.0)

        with XDMFFile(self.mesh.comm, "out_darcy/u.xdmf","w") as file:
            file.write_mesh(self.mesh)
            file.write_function(vel_f, 0.0)


class Projector():
    def __init__(self, V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = inner(u, v) * ufl.dx
        # self.V = V
        self.u = dfx.fem.Function(V) # Create function
        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)

    def __call__(self, f):
        v = ufl.TestFunction(self.u.function_space)
        L = inner(f, v) * ufl.dx
        
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters) # Compile form
        self.A = assemble_matrix(self.a_cpp, bcs=[])
        
        self.A.assemble()

        self.b = create_vector(self.L_cpp) # Create RHS vector

        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setOperators(self.A)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverType('mumps')
        self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization
        with self.b.localForm() as b_loc: b_loc.set(0)

        # Assemble vector and set BCs
        assemble_vector(self.b, self.L_cpp)
        # apply_lifting(self.b, [self.a_cpp], bcs = [])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # MPI communication
        # set_bc(self.b, bcs=[])
        
        self.solver.solve(self.b, self.u.x.petsc_vec)

        # Destroy PETSc linear algebra objects and solver
        self.solver.destroy()
        self.A.destroy()
        self.b.destroy()

        return self.u

if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True

    T = 1.5 # Final simulation time
    dt = 0.01 # Timestep size
    N = 10 # Number of mesh cells
    # N = int(argv[1]) # Number of mesh cells
    D_value = 1e-2 # Diffusion coefficient
    k = 1 # Finite element polynomial degree

    # Create transport solver object
    perfusion_sim = PerfusionSolver(N_cells=N,
                                    D_value=D_value,
                                    element_degree=k,
                                    write_output=write_output)
    perfusion_sim.setup()