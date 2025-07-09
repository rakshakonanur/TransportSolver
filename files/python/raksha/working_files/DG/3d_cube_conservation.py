from pyclbr import Function
import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad, div, inner, SpatialCoordinate, conditional
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print = PETSc.Sys.Print

# MESHTAGS
OUTLET = 1
INLET = 2
LEFT = 3
RIGHT = 4
TOP = 5
BOTTOM = 6
WALL = 7


# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

# Generate mesh
def create_cube_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, nx = N_cells, ny = N_cells, nz = N_cells,
                                           cell_type = dfx.mesh.CellType.tetrahedron,
                                           ghost_mode = dfx.mesh.GhostMode.shared_facet)
        
        inlet = np.array([0.2, 0.5, 0.5]) # Inlet point
        outlet = np.array([0.5, 0.5, 0.5]) # Outlet point

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], 1.0)
        def wall(x):
            return np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        internal_indices, internal_markers = [], []
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

        internal_inlet_facets = find_interior_facets_near_point(mesh, inlet, tol=0.05)
        internal_outlet_facets = find_interior_facets_near_point(mesh, outlet, tol=0.05)

        internal_indices = np.concatenate([internal_inlet_facets, internal_outlet_facets])
        internal_markers = np.concatenate([
            np.full_like(internal_inlet_facets, INLET),
            np.full_like(internal_outlet_facets, OUTLET)
        ])
 
        sorted_internal = np.argsort(internal_indices)

        external_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])
        internal_tags = dfx.mesh.meshtags(mesh, fdim, internal_indices[sorted_internal], internal_markers[sorted_internal])
        print("Tagged facets:", external_tags.indices)
        print("Tag values:   ", external_tags.values)
        print("Inlet facets: ", internal_tags.find(INLET))
        print("Outlet facets:", internal_tags.find(OUTLET))

        return mesh, external_tags, internal_tags

def find_interior_facets_near_point(mesh, point, tol=0.01):
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    num_facets = mesh.topology.index_map(fdim).size_local

    interior_facets = []
    for facet in range(num_facets):
        connected_cells = mesh.topology.connectivity(fdim, mesh.topology.dim).links(facet)
        if len(connected_cells) == 2:
            x = dfx.mesh.compute_midpoints(mesh, fdim, np.array([facet], dtype=np.int32))
            if np.linalg.norm(x[0] - point) < tol:
                interior_facets.append(facet)
    return np.array(interior_facets, dtype=np.int32)


class Transport:
     
    def __init__(self, N_cells: int,
                       T: float,
                       dt: float,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.N = N_cells
        self.mesh, self.external_tags, self.internal_tags = create_cube_mesh_with_tags(N_cells=N_cells)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output

        # Temporal parameters
        self.T = T
        self.dt = dt
        self.t = 0
        self.num_timesteps = int(T / dt)

    def setup(self):
        """ Set up the problem. """

        # Simulation parameters
        self.D_value = 1e-3
        self.k = 1 # Element degree
        self.t = 0 # Initial time
        self.T = 10 # Final time
        self.dt = 0.1 # Timestep size
        self.num_timesteps = int(self.T / self.dt)
        self.n = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure("dx", domain=self.mesh) # Cell integrals
        self.dS_inlet = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.internal_tags) # Interior cell integrals
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.external_tags) # Exterior facet integrals
        self.dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.internal_tags) # Interior facet integrals

        # Function spaces
        Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
        V = dfx.fem.functionspace(self.mesh, Pk_vec) # function space for velocity
        self.u = dfx.fem.Function(V) # velocity
        Pk = element("Lagrange", self.mesh.basix_cell(), degree=1)
        self.u.x.array[:] = 0.0 # Set velocity u=0.0 everywhere
        self.u.x.array[0::3] = 0.1 # Set velocity u=0.1 in x-direction
        DG = element("DG", self.mesh.basix_cell(), degree=self.element_degree) # Discontinuous Galerkin element
        W = dfx.fem.functionspace(self.mesh, DG) # function space for concentration
        deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Time step size

        print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

        # === Trial, test, and solution functions ===
        c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
        self.c_h = dfx.fem.Function(W) # concentration at current time step
        self.c_ = dfx.fem.Function(W) # concentration at previous time step

        self.bc_func = dfx.fem.Function(W) # Boundary condition function
        self.bc_func.interpolate(lambda x: 1 + np.zeros_like(x[0])) # Set initial condition at inlet
        # np.array((1.0,) * self.mesh.geometry.dim, dtype= dfx.default_scalar_type)

        # # self.u_ex = lambda x: 0 + np.zeros_like(x[0]) #x[0]**2 + 2*x[1]**2
        # self.x = ufl.SpatialCoordinate(self.mesh)
        # # self.g = dot(self.n, grad(self.u)) # corresponding to the Neumann BC
        # self.g = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Neumann BC at outlet
        # L_neumann = inner(self.g, w) * self.ds(OUTLET)

        # print(f"Boundary condition: Neumann on facet {OUTLET} with value {self.g}", flush=True)

        #------VARIATIONAL FORM------#
        self.D = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.D_value)) # Diffusion coefficient
        f = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # SIPG penalty parameter

        u_mag = ufl.sqrt(ufl.dot(self.u, self.u) + 1e-10) # Magnitude of velocity
        Pe = u_mag * hf / (2 * self.D) # Peclet number
        beta = (ufl.cosh(Pe)/ufl.sinh(Pe))- (1/Pe)
        tau = hf * beta/ (2 * u_mag + 1e-10) # Stabilization parameter
        nitsche = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10.0)) # Nitsche parameter

        # Variational form
        a_time     = c * w / deltaT * self.dx
        a_advect   = - dot(c*self.u, grad(w)) * self.dx # Advection term
        a_diffuse  = dot(self.D * grad(c), grad(w)) * self.dx

        a = a_time + a_advect + a_diffuse
        L = (self.c_ / deltaT + f) * w * self.dx

        # Diffusive terms with interior penalization
        a  += self.D('+') * alpha('+') / hf('+') * dot(jump(w, self.n), jump(c, self.n)) * self.dS
        a  -= self.D('+') * dot(avg(grad(w)), jump(c, self.n)) * self.dS
        a  -= self.D('+') * dot(jump(w, self.n), avg(grad(c))) * self.dS

        # SUPG terms
        # residual = dot(self.u, grad(self.c_h)) - self.D * div(grad(self.c_h)) + (self.c_h - self.c_) / deltaT - f

        # Impose BC using Nitsche's method
        a_nitsche = nitsche / avg(hf) * avg(c) * avg(w) * self.dS(INLET)
        L_nitsche = nitsche / avg(hf) * avg(self.bc_func) * avg(w) * self.dS(INLET)

        # Upwind velocity
        un = (dot(self.u, self.n) + abs(dot(self.u, self.n))) / 2.0
        a_upwind = dot(jump(w), un('+') * c('+') - un('-') * c('-')) * self.dS

        # Enforce Neumann BC at the outlet
        # u_n = dot(self.u, self.n)
        # c_ext = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0)) # Set external concentration to zero
        # L_outflow = - conditional(avg(un) > 0, un('+') * c_ext * avg(w), 0.0) * self.dS(OUTLET)

        outflux  = c('+')*dot(self.u('+'), self.n('+')) # Only advective flux on outflow boundary, diffusive flux is zero
        u_normal = dot(self.u, self.n) # The normal velocity

        # Create conditional expressions
        cond  = ufl.lt(u_normal, 0.0) # Condition: True if the normal velocity is less than zero, u.n < 0
        minus = ufl.conditional(cond, 1.0, 0.0) # Conditional that returns 1.0 if u.n <  0, else 0.0. Used to "activate" terms on the influx  boundary
        plus  = ufl.conditional(cond, 0.0, 1.0) # Conditional that returns 1.0 if u.n >= 0, else 0.0. Used to "activate" terms on the outflux boundary
        
        # Add outflux term to the weak form
        a += plus('+')*outflux* w('+') * self.dS(OUTLET)

        a += a_nitsche + a_upwind # + a_outflow
        L += L_nitsche # + L_outflow

        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters)

        # No strong Dirichlet BCs
        self.bcs = []

        # self.c_.interpolate(self.bc_func) # Previous timestep
        self.c_.x.array[:] = 0.0

        # Create output function in P1 space
        P1 = element("Lagrange", self.mesh.basix_cell(), degree=1) # Linear Lagrange elements
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, P1))
        self.c_out.interpolate(self.bc_func)
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk_vec))

        # Interpolate it into the velocity function
        self.u_out.x.array[:] = self.u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c = dfx.fem.form(self.c_h * self.dx)
        # self.error_form = residual**2 * self.dx # calculates square of L2 error over the interior facets

        # === Linear system ===
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

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/cube_conc_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.c_out.interpolate(self.c_h)  # Interpolate the concentration function
            self.xdmf_c.write_function(self.c_out, self.t)

            out_str = './output/cube_vel_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_u = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_u.write_mesh(self.mesh)
            self.xdmf_u.write_function(self.u_out, self.t)

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

            # error_squared = dfx.fem.assemble_scalar(dfx.fem.form(self.error_form)) # assemble into a scalar, by converting symbolic UFL form to Fenicsx
            # total_residual = self.mesh.comm.allreduce(error_squared, op=MPI.SUM) # gather all the errors from all processes in case of parallel execution
            # print(f"Total residual: {np.sqrt(total_residual):.2e}")


            if self.write_output:
                # Write to file
                self.c_out.interpolate(self.c_h)  # Interpolate the concentration function
                self.xdmf_c.write_function(self.c_out, self.t)

                self.u_out.interpolate(self.u)
                self.xdmf_u.write_function(self.u_out, self.t)
        
        # fig, ax = plt.subplots()
        # line, = ax.plot(x_coords, snapshots[0])
        # ax.set_ylim(min(map(np.min, snapshots)), max(map(np.max, snapshots))+0.1)
        # ax.set_xlabel("x")
        # ax.set_ylabel("Concentration")
        # ax.set_title("Concentration evolution")

        # def update(frame):
        #     line.set_ydata(snapshots[frame])
        #     ax.set_title(f"Time: {time_values[frame]:2.2f}")
        #     return line,

        # ani = FuncAnimation(fig, update, frames=len(snapshots), interval=10, blit=False, repeat=False)

        # # Plot the solution
        # try:
        #     import matplotlib.pyplot as plt
        # except:
        #     RuntimeError("A matplotlib is required to plot the solution.")
        # plt.plot(self.c_h.function_space.tabulate_dof_coordinates()[:, 0], self.c_h.x.array)
        # plt.show()

        # Save as MP4 (requires ffmpeg installed)
        # ani.save("hal_conservation_0.1.mp4", writer="ffmpeg", fps=20)

        # Or save as GIF (requires imagemagick installed)
        # ani.save("sine_wave.gif", writer="imagemagick", fps=20)

        plt.close()


if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True

    T = 1.5 # Final simulation time
    dt = 0.01 # Timestep size
    N = int(argv[1]) # Number of mesh cells
    D_value = 1e-2 # Diffusion coefficient
    k = 2 # Finite element polynomial degree

    # Create transport solver object
    transport_sim = Transport(N_cells=N,
                                    T=T,
                                    dt=dt,
                                    D_value=D_value,
                                    element_degree=k,
                                    write_output=write_output)
    transport_sim.setup()
    transport_sim.run()