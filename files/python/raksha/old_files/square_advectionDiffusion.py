import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting

print = PETSc.Sys.Print

# MESHTAGS
LEFT    = 1
RIGHT   = 2
BOT     = 3
TOP     = 4

# Set compiler options for runtime optimization
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def create_square_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N_cells, N_cells,
                                           cell_type = dfx.mesh.CellType.triangle,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], 1.0)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left) # locates the left boundary
        bc_facet_indices.append(inlet_BC_facets) # adds the facets from left boundary
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT)) # assigns the tag LEFT (1) to the left boundary facets in same order

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32) # combines into a single array
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32) # combines into a single array

        sorted_facets = np.argsort(bc_facet_indices) # sorts the facets in ascending order

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets]) 
        # creates a meshtag object with the sorted facets and their corresponding tags

        return mesh, facet_tags

class TransportSolver:

    def __init__(self, N_cells: int,
                       T: float,
                       dt: float,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.N = N_cells
        self.mesh, self.ft = create_square_mesh_with_tags(N_cells=N_cells)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output

        # Temporal parameters
        self.T = T
        self.dt = dt
        self.t = 0
        self.num_timesteps = int(T / dt)

    def setup(self):
        ''' Setup of the variational problem for the advection-diffusion equation
            dc/dt + div(J) = f
            with
            J = c*u - D*grad(c)
            where
                - c is solute concentration
                - u is velocity
                - D is diffusion coefficient
                - f is a source term
        '''

        # Facet normal and integral measures
        n  = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.ft) # Exterior facet integrals
        dS = ufl.Measure('dS', domain=self.mesh) # Interior facet integrals

        # Finite elements for the velocity and the concentration
        P2_vec = element("Lagrange", self.mesh.basix_cell(), degree=2, shape=(self.mesh.geometry.dim,)) # Quadratic vector Lagrange elements
        V = dfx.fem.functionspace(self.mesh, P2_vec) # Velocity function space
        u = dfx.fem.Function(V) # velocity
        velocity_scale = 0.1
        u.interpolate(lambda x: (-x[0], x[1])) # div(u)=0 by construction
        u.x.array[:] *= velocity_scale # Scale the velocity magnitude
        P1 = element("Lagrange", self.mesh.basix_cell(), degree=1) # Linear Lagrange elements
        DG = element("DG", self.mesh.basix_cell(), degree=self.element_degree) # DG elements

        # Function spaces
        W = dfx.fem.functionspace(self.mesh, DG) # Concentration function space

        # Trial and test functions
        c, w = ufl.TrialFunction(W), ufl.TestFunction(W)

        # Functions for storing solution
        self.c_h  = dfx.fem.Function(W) # Concentration at current  timestep
        self.c_   = dfx.fem.Function(W) # Concentration at previous timestep

        #------VARIATIONAL FORM------#
        D = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.D_value)) # Diffusion coefficient
        f = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term 
        deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Form-compiled timestep        
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # SIPG penalty parameter
        beta  = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10.0)) # Nitsche penalty parameter

        print("Total number of dofs: ", W.dofmap.index_map.size_global, flush=True)

        # Dirichlet BC for the right side of the square
        self.bc_func = dfx.fem.Function(W)
        self.bc_func.interpolate(lambda x: np.abs(np.sin(np.pi*3*x[1]))) # set sinusoidal BC for right side of the square

        # Upwind velocity
        un = (dot(u, n) + abs(dot(u, n))) / 2.0

        # Bilinear form
        a0 = c * w / deltaT * self.dx # Time derivative
        a1 = dot(grad(w), D * grad(c) - u * c) * self.dx # Flux term integrated by parts

        # Diffusive terms with interior penalization
        a2  = D('+') * alpha('+') / hf('+') * dot(jump(w, n), jump(c, n)) * dS
        a2 -= D('+') * dot(avg(grad(w)), jump(c, n)) * dS
        a2 -= D('+') * dot(jump(w, n), avg(grad(c))) * dS

        # Advection terms upwinded
        a3 = dot(jump(w), un('+') * c('+') - un('-') * c('-')) * dS

        # Weakly enforced BC on the right-hand side of the square, using Nitsche's method
        a4 = beta / hf * c*w * self.ds(RIGHT)

        # Sum all bilinear form terms and compile form
        a = a0+a1+a2+a3+a4
        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)

        # Linear form
        L = self.c_ * w / deltaT * self.dx + w * f * self.dx # Time derivative term + source term
        L += beta / hf * self.bc_func*w * self.ds(RIGHT) # Weakly enforce Dirichlet BC using Nitsche's method

        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters) # Compile form
        
        # No strong Dirichlet BCs
        self.bcs = []

        self.c_.interpolate(self.bc_func) # Previous timestep

        # Create output function in P1 space
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, P1))
        self.c_out.interpolate(self.bc_func)

        # Compile total concentration integral for calculation in time-loop
        self.total_c = dfx.fem.form(self.c_h * self.dx)

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/square_concentration_D=' + f'{D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.xdmf_c.write_function(self.c_out, self.t)

            # Write velocity to file
            vtx_u = dfx.io.VTXWriter(MPI.COMM_WORLD, './output/velocity.bp', [u], 'BP4')
            vtx_u.write(0)
            vtx_u.close()

        # Assemble linear system
        self.A = assemble_matrix(self.a_cpp, bcs=self.bcs)
        self.A.assemble()

        self.b = create_vector(self.L_cpp) # Create RHS vector

        # Configure direct solver
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverType('mumps')
        self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization
     

    def assemble_transport_RHS(self):
        """ Assemble the right-hand side of the variational problem. """
    
        # Zero entries to avoid accumulation
        with self.b.localForm() as b_loc: b_loc.set(0)

        # Assemble vector and set BCs
        assemble_vector(self.b, self.L_cpp)
        apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # MPI communication
        set_bc(self.b, bcs=self.bcs)

    def run(self):
        """ Run transport simulations. """

        for _ in range(self.num_timesteps):

            self.t += self.dt

            self.assemble_transport_RHS()

            # Compute solution to the advection-diffusion equation and perform parallel communication
            self.solver.solve(self.b, self.c_h.x.petsc_vec)
            self.c_h.x.scatter_forward()

            # Update previous timestep
            self.c_.x.array[:] = self.c_h.x.array.copy()

            # Print stuff
            print(f"Timestep t = {self.t}")

            print("Maximum concentration: ", self.mesh.comm.allreduce(self.c_h.x.array.max(), op=MPI.MAX))
            print("Minimum concentration: ", self.mesh.comm.allreduce(self.c_h.x.array.min(), op=MPI.MIN))

            total_c = dfx.fem.assemble_scalar(self.total_c)
            total_c = self.mesh.comm.allreduce(total_c, op=MPI.SUM)
            print(f"Total concentration: {total_c:.2e}")

            if self.write_output:
                # Write to file
                self.c_out.interpolate(self.c_h)
                self.xdmf_c.write_function(self.c_out, self.t)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True

    T = 1.5 # Final simulation time
    dt = 0.01 # Timestep size
    N = int(argv[1]) # Number of mesh cells
    D_value = 1e-2 # Diffusion coefficient
    k = 2 # Finite element polynomial degree

    # Create transport solver object
    transport_sim = TransportSolver(N_cells=N,
                                    T=T,
                                    dt=dt,
                                    D_value=D_value,
                                    element_degree=k,
                                    write_output=write_output)
    transport_sim.setup()
    transport_sim.run()