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
from dolfinx.io import XDMFFile, VTKFile
import adios4dolfinx

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def import_mesh(xdmf_file):
    """
    Import a mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        mesh_facets = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, mesh_facets

class PerfusionSolver:
    def __init__(self, xdmf_file: str):
        """
        Initialize the PerfusionSolver with a given STL file and branching data.
        """
        self.D_value = 1e-2
        self.element_degree = 1
        self.write_output = True
        self.mesh, self.facets = import_mesh(xdmf_file)

    def setup(self):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        
        # k = self.element_degree
        k = 1
        P_el = element("Lagrange", self.mesh.basix_cell(), k)
        u_el = element("DG", self.mesh.basix_cell(), k-1, shape=(self.mesh.geometry.dim,))

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
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        bc_wall = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        wall_BC_facets = dfx.mesh.exterior_facet_indices(self.mesh.topology)
        wall_BC_dofs = dfx.fem.locate_dofs_topological(W, fdim, wall_BC_facets)

        self.bc_func = dfx.fem.Function(W)
        self.bc_func.x.array[:] = 1.0  # * 1333.22  # Convert mmHg to Pa
        dofs = self.facets.find(1) # Tag 1 is the outlets of the branched network
        bcs = [dfx.fem.dirichletbc(self.bc_func, dofs),
               dfx.fem.dirichletbc(bc_wall, np.setdiff1d(wall_BC_dofs, dofs), W)]
        
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

        vtkfile = VTKFile(MPI.COMM_WORLD, "u.vtu", "w")

        # Write the function to the VTK file
        vtkfile.write_function(vel_f)

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
        
        
if __name__ == "__main__":
    solver = PerfusionSolver("../geometry/vertex_tags_nearest.xdmf")
    solver.setup()

    # Further processing can be added here