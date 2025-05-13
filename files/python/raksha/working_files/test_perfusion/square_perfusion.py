import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx import (fem, io, mesh)
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting, LinearProblem
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

"""
    From the DOLFINx tutorial: Mixed formulation of the Poisson equation
    https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_mixed-poisson.html
"""

print = PETSc.Sys.Print

# MESHTAGS
LEFT    = 1
RIGHT   = 2
BOT     = 3
TOP     = 4

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def create_rectangle_mesh_with_tags(L: float, H: float, N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]),np.array([L, H])], [N_cells,N_cells],
                                           cell_type = dfx.mesh.CellType.quadrilateral,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], L)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], H)

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
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags

class PerfusionSolver:

    def __init__(self, H : float,
                       L : float,
                       N_cells: int,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.H = H
        self.L = L
        self.N = N_cells
        self.mesh, self.ft = create_rectangle_mesh_with_tags(H, L, N_cells=N_cells)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output

    def setup(self):
        ''' Setup the solver. '''
        fdim = self.mesh.topology.dim - 1  
        k = self.element_degree
        Q_el = element("BDMCF", self.mesh.basix_cell(), k)
        P_el = element("DG", self.mesh.basix_cell(), k - 1)
        V_el = mixed_element([Q_el, P_el])
        V = fem.functionspace(self.mesh, V_el)
        beta  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # Nitsche penalty parameter
        hf = ufl.CellDiameter(self.mesh) # Cell diameter

        (sigma, u) = TrialFunctions(V)
        (tau, v) = TestFunctions(V)

        x = SpatialCoordinate(self.mesh)
        # f = 0.0 * exp(-((x[0] - self.L/2) * (x[0] - self.L/2) + (x[1] - self.H/2) * (x[1] - self.H/2)) / 0.05)
        f = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term 

        # facets_top = mesh.locate_entities_boundary(self.mesh, fdim, TOP)
        Q, _ = V.sub(0).collapse()
        P,_ = V.sub(1).collapse()
        dofs_top = fem.locate_dofs_topological((V.sub(0), Q), fdim, self.ft.find(TOP))

        # Dirichlet BC for the right side of the square
        self.bc_func_right = dfx.fem.Function(P)
        self.bc_func_right.interpolate(lambda x: np.full(x.shape[1], 0.0))

        self.bc_func_left = dfx.fem.Function(P)
        self.bc_func_left.interpolate(lambda x: np.full(x.shape[1], 100.0))

        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.ft) # Exterior facet integrals
        dx = Measure("dx", self.mesh)
        a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
        # Weakly enforced BC on the right-hand side of the square, using Nitsche's method
        a4 = beta / hf * u * v * self.ds(RIGHT)
        a4 += beta / hf * u * v * self.ds(LEFT)
        a += a4

        L = -inner(f, v) * dx
        L += beta / hf * self.bc_func_right* v * self.ds(RIGHT) 
        L += beta / hf * self.bc_func_left* v * self.ds(LEFT) # Weakly enforce Dirichlet BC using Nitsche's method

        # def f1(x):
        #     values = np.zeros((2, x.shape[1]))
        #     values[1, :] = np.sin(5 * x[0])
        #     return values


        # f_h1 = fem.Function(Q)
        # f_h1.interpolate(f1)
        # bc_top = fem.dirichletbc(f_h1, dofs_top, V.sub(0))


        # # facets_bottom = mesh.locate_entities_boundary(self.mesh, fdim, BOT)
        # dofs_bottom = fem.locate_dofs_topological((V.sub(0), Q), fdim, self.ft.find(BOT))


        # def f2(x):
        #     values = np.zeros((2, x.shape[1]))
        #     values[1, :] = -np.sin(5 * x[0])
        #     return values


        # f_h2 = fem.Function(Q)
        # f_h2.interpolate(f2)
        # bc_bottom = fem.dirichletbc(f_h2, dofs_bottom, V.sub(0))


        # bcs = [bc_top, bc_bottom]
        self.bcs = []

        problem = LinearProblem(a, L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
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

        with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/u.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            file.write_function(u_h)
        
if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True

    T = 1.5 # Final simulation time
    dt = 0.01 # Timestep size
    L = 1.0 # Length of the domain
    H = 2.0 # Height of the domain
    N = 64 # Number of mesh cells
    # N = int(argv[1]) # Number of mesh cells
    D_value = 1e-2 # Diffusion coefficient
    k = 1 # Finite element polynomial degree

    # Create transport solver object
    perfusion_sim = PerfusionSolver(L=L,
                                    H=H,
                                    N_cells=N,
                                    D_value=D_value,
                                    element_degree=k,
                                    write_output=write_output)
    perfusion_sim.setup()

