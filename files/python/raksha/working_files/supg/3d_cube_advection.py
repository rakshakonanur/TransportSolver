from pyclbr import Function
import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad, div, inner, SpatialCoordinate
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
OUTLET = 0
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

# Generate mesh
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
        self.mesh, self.ft = create_cube_mesh_with_tags(N_cells=N_cells)
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

        # Function spaces
        Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
        V = dfx.fem.functionspace(self.mesh, Pk_vec) # function space for velocity
        self.u = dfx.fem.Function(V) # velocity
        Pk = element("Lagrange", self.mesh.basix_cell(), degree=1)
        self.u.x.array[:] = 0.0 # Set velocity u=0.0 everywhere
        self.u.x.array[0::3] = 0.10 # Set velocity u=1.0 in x-direction
        W = dfx.fem.functionspace(self.mesh, Pk) # function space for concentration
        deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Time step size

        print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

        # === Trial, test, and solution functions ===
        c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
        self.c_h = dfx.fem.Function(W) # concentration at current time step
        self.c_ = dfx.fem.Function(W) # concentration at previous time step

        # === Boundary conditions ===
        fdim = self.mesh.topology.dim - 1
        inlet = np.array([0.2, 0.5, 0.5]) # Inlet point
        outlet = np.array([0.8, 0.5, 0.5]) # Outlet point
        # self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)

        dof_inlet = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], inlet[0]) & np.isclose(x[1], inlet[1]) & np.isclose(x[2], inlet[2]))

        bc_left = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1.0))
        self.bcs = [dfx.fem.dirichletbc(bc_left, dof_inlet, W)]  # Only apply at inlet

        def near_outlet(x):
            return np.isclose(x[0], outlet[0]) & np.isclose(x[1], outlet[1]) & np.isclose(x[2], outlet[2])
        outlet_facets = dfx.mesh.locate_entities_boundary(self.mesh, fdim, near_outlet)

        self.facet_tags = dfx.mesh.meshtags(self.mesh, fdim, outlet_facets, np.full_like(outlet_facets, OUTLET))
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)

        # self.u_ex = lambda x: 0 + np.zeros_like(x[0]) #x[0]**2 + 2*x[1]**2
        self.x = ufl.SpatialCoordinate(self.mesh)
        # self.g = dot(self.n, grad(self.u)) # corresponding to the Neumann BC
        self.g = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Neumann BC at outlet
        L_neumann = inner(self.g, w) * self.ds(OUTLET)

        print(f"Boundary condition: Neumann on facet {OUTLET} with value {self.g}", flush=True)

        # bc_left = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(1.0))
        # dof_left = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0.0))
        # self.bcs = [dfx.fem.dirichletbc(bc_left, dof_left, W)]  # Only apply at inlet

        # bc_right = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0))
        # dof_right = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 1.0))
        # self.bcs.append(dfx.fem.dirichletbc(bc_right, dof_right, W))  # Apply at outlet

        #------VARIATIONAL FORM------#
        self.D = dfx.fem.Constant(self.mesh, self.D_value) # Diffusion coefficient
        f = dfx.fem.Constant(self.mesh, 0.0) # Source term
        hf = ufl.CellDiameter(self.mesh) # Cell diameter

        u_mag = ufl.sqrt(ufl.dot(self.u, self.u) + 1e-10) # Magnitude of velocity
        Pe = u_mag * hf / (2 * self.D) # Peclet number
        beta = (ufl.cosh(Pe)/ufl.sinh(Pe))- (1/Pe)
        tau = hf * beta/ (2 * u_mag + 1e-10) # Stabilization parameter

        # Variational form
        a_time     = c * w / deltaT * self.dx
        a_advect   = dot(self.u, grad(c)) * w * self.dx
        a_diffuse  = dot(grad(c), grad(w)) * self.D * self.dx

        a = a_time + a_advect + a_diffuse
        L = (self.c_ / deltaT + f) * w * self.dx

        # SUPG terms
        residual = dot(self.u, grad(self.c_h)) - self.D * div(grad(self.c_h)) + (self.c_h - self.c_) / deltaT - f
        v_supg = tau * dot(self.u, grad(w))

        a_supg = v_supg * (c / deltaT + dot(self.u, grad(c)) - self.D * div(grad(c))) * self.dx
        L_supg = v_supg * (self.c_ / deltaT + f) * self.dx

        a += a_supg
        L += L_supg
        L += L_neumann

        self.a_cpp = dfx.fem.form(a)
        self.L_cpp = dfx.fem.form(L)

        # Create output function in P1 space
        P1 = element("Lagrange", self.mesh.basix_cell(), degree=1) # Linear Lagrange elements
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, P1))
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk_vec))

        # Interpolate it into the velocity function
        self.u_out.x.array[:] = self.u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c = dfx.fem.form(self.c_h * self.dx)
        self.error_form = residual**2 * self.dx # calculates square of L2 error over the interior facets

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

            error_squared = dfx.fem.assemble_scalar(dfx.fem.form(self.error_form)) # assemble into a scalar, by converting symbolic UFL form to Fenicsx
            total_residual = self.mesh.comm.allreduce(error_squared, op=MPI.SUM) # gather all the errors from all processes in case of parallel execution
            print(f"Total residual: {np.sqrt(total_residual):.2e}")


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

    T = 5.0 # Final simulation time
    dt = 0.05 # Timestep size
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