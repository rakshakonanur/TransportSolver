import ufl
import numpy as np
import dolfinx as dfx

from ufl import dot, grad, jump
from sys import argv
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx.mesh import locate_entities_boundary, meshtags, create_interval
from dolfinx.fem import Function, Constant, dirichletbc, locate_dofs_geometrical, functionspace, form, assemble_scalar

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TransportSolver:

    def __init__(self, L : float,
                       u_val : float,
                       c_val,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.L = L
        self.u_val = u_val
        self.D_value = 1e-3
        self.element_degree = 1
        self.write_output = write_output
        # self.N = int((10* u_val * L) / (2 * self.D_value)) # Number of mesh cells: based on stability criterion of grid Pe
        self.N = 10
        self.c_val = c_val
        self.mesh = create_interval(MPI.COMM_WORLD, self.N, [0.0, L])
        self.mesh_tagging()

        # Temporal parameters
        self.T = 10
        self.dt = 0.1
        self.t = 0
        self.num_timesteps = int(self.T / self.dt)

    def mesh_tagging(self):
        fdim = self.mesh.topology.dim - 1

        def left_boundary(x): return np.isclose(x[0], 0.0)
        def right_boundary(x): return np.isclose(x[0], 1.0)

        left_facets = locate_entities_boundary(self.mesh, fdim, left_boundary)
        right_facets = locate_entities_boundary(self.mesh, fdim, right_boundary)

        facet_indices = np.concatenate([left_facets, right_facets])
        facet_markers = np.concatenate([np.full(len(left_facets), 1, dtype=np.int32),
                                        np.full(len(right_facets), 2, dtype=np.int32)])
        self.facet_tag = meshtags(self.mesh, fdim, facet_indices, facet_markers)
        print("Facet indices: ", facet_indices, flush=True)
        print("Facet markers: ", facet_markers, flush=True)

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
        # self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tag)
        dS = ufl.Measure("dS", domain=self.mesh)

        # === Function spaces ===
        Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=self.element_degree, shape=(self.mesh.geometry.dim,))
        V = functionspace(self.mesh, Pk_vec)
        u = Function(V)
        u.x.array[:] = self.u_val  # constant velocity field

        Pk = element("Lagrange", self.mesh.basix_cell(), degree=self.element_degree)
        self.W = functionspace(self.mesh, Pk)
        print("Total number of concentration dofs: ", self.W.dofmap.index_map.size_global, flush=True)

        # === Trial, test, and solution functions ===
        c, w = ufl.TrialFunction(self.W), ufl.TestFunction(self.W)
        self.c_h = Function(self.W)
        self.c_ = Function(self.W)

        #------VARIATIONAL FORM------#
        D = Constant(self.mesh, dfx.default_scalar_type(self.D_value))
        f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        h = Constant(self.mesh, dfx.default_scalar_type(1.0))
        deltaT = Constant(self.mesh, dfx.default_scalar_type(self.dt))
        beta = Constant(self.mesh, dfx.default_scalar_type(10.0))
        hf = ufl.CellDiameter(self.mesh)


        print("Total number of dofs: ", self.W.dofmap.index_map.size_global, flush=True)

        # === Boundary conditions ===

        self.bc_left_func = Function(self.W)
        self.bc_left_func.x.array[:] = self.c_val 

        dof_left = locate_dofs_geometrical(self.W, lambda x: np.isclose(x[0], 0.0))
        self.bcs = [dirichletbc(self.bc_left_func, dof_left)]

        # # For constant boundary condition
        # self.bc_left = Constant(self.mesh, dfx.default_scalar_type(self.c_val[0]))
        # self.dof_left = locate_dofs_geometrical(self.W, lambda x: np.isclose(x[0], 0.0))
        # # self.bcs = [dirichletbc(self.bc_left, self.dof_left, self.W)]  # Only apply at inlet

        # self.bc_right = Constant(self.mesh, dfx.default_scalar_type(0.0))
        # self.dof_right = locate_dofs_geometrical(self.W, lambda x: np.isclose(x[0], self.L))
        # self.bcs = [dirichletbc(self.bc_left, self.dof_left, self.W), dirichletbc(self.bc_right, self.dof_right, self.W)]  # Only apply at inlet

        # === Variational Form ===
        un = (dot(u, n) + abs(dot(u, n))) / 2.0

        a_time     = c * w / deltaT * ufl.dx
        a_advect   = dot(u, grad(c)) * w * ufl.dx
        a_diffuse  = dot(grad(c), grad(w)) * D * ufl.dx

        a = a_time + a_advect + a_diffuse
        L = (self.c_ / deltaT + f) * w * ufl.dx

        self.a_cpp = form(a)
        self.L_cpp = form(L)

        # Create output function in P1 space
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk))

        # === Total concentration integral ===
        self.total_c_form = form(self.c_h * ufl.dx)

        # === Linear system ===
        self.A = assemble_matrix(self.a_cpp, bcs=self.bcs)
        self.A.assemble()
        self.b = create_vector(self.L_cpp)

        # if self.write_output:
        #     # Create output file for the concentration
        #     out_str = './output/square_concentration_D=' + f'{D.value}' + '.xdmf'
        #     self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
        #     self.xdmf_c.write_mesh(self.mesh)
        #     self.xdmf_c.write_function(self.c_out, self.t)

        #     # Write velocity to file
        #     vtx_u = dfx.io.VTXWriter(MPI.COMM_WORLD, './output/velocity.bp', [u], 'BP4')
        #     vtx_u.write(0)
        #     vtx_u.close()

        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverType('mumps')
        self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1)

        # === RHS assembler ===
    def assemble_transport_RHS(self):
        with self.b.localForm() as b_loc: b_loc.set(0)
        assemble_vector(self.b, self.L_cpp)
        apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, bcs=self.bcs)

    def run(self):
        """ Run transport simulations. """

        # Allocate list to store time series snapshots
        self.snapshots = []
        self.time_values = []

      
        for _ in range(self.num_timesteps):
            
            self.t += self.dt
            # Get x positions once (only for Lagrange P1 elements in 1D)
            self.x_coords = self.c_h.function_space.tabulate_dof_coordinates()[:, 0]

            val = self.c_val[_-1]
            self.bc_left = val
            self.bcs = [dirichletbc(self.bc_left, self.dof_left, self.W)]  # Only apply at inle

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

            total_c = dfx.fem.assemble_scalar(self.total_c_form)
            total_c = self.mesh.comm.allreduce(total_c, op=MPI.SUM)
            print(f"Total concentration: {total_c:.2e}")

            # if self.write_output:
            #     # Write to file
            #     self.c_out.interpolate(self.c_h)
            #     self.xdmf_c.write_function(self.c_out, self.t)

            self.snapshots.append(self.c_h.x.array.copy())
            self.time_values.append(self.t)
            

        fig, ax = plt.subplots()
        line, = ax.plot(self.x_coords, self.snapshots[0])
        ax.set_ylim(min(map(np.min, self.snapshots)), max(map(np.max, self.snapshots))+0.1)
        ax.set_xlabel("x")
        ax.set_ylabel("Concentration")
        ax.set_title("Concentration evolution")

        def update(frame):
            line.set_ydata(self.snapshots[frame])
            ax.set_title(f"Time: {self.time_values[frame]:2.2f}")
            return line,
        
        

        ani = FuncAnimation(fig, update, frames=len(self.snapshots), interval=10, blit=False, repeat=False)
        # c_true = ((np.exp(self.u_val * self.x_coords) / self.D_value) - np.exp((self.u_val * self.L) / self.D_value)) / (1 - np.exp((self.u_val * self.L) / self.D_value))
        Pe = self.u_val / self.D_value
        c_true = (np.exp(Pe * self.x_coords) - np.exp(Pe * self.L)) / (1 - np.exp(Pe * self.L))
        plt.plot(self.x_coords, c_true, label="Analytical solution")
        # plt.plot(self.c_h.function_space.tabulate_dof_coordinates()[:, 0], self.c_h.x.array)
        plt.legend()
        plt.show()

        # ani.save("raksha_steady_0.01_t10.mp4", writer="ffmpeg", fps=20)
        return self.snapshots, self.time_values


# === Time loop ===
if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True
    L = 1.0
    u_val = 0.01 # Velocity value
    k = 1 # Finite element polynomial degree

    # Create transport solver object
    transport_sim = TransportSolver(L=L,
                                    c_val=np.full(1000, 1.0),
                                    # c_val=np.linspace(1, 1.5, 100),
                                    u_val=u_val,
                                    element_degree=k,
                                    write_output=write_output)
    transport_sim.setup()
    transport_sim.run()

#     # Allocate list to store time series snapshots
#     snapshots = []
#     time_values = []

#     # Get x positions once (only for Lagrange P1 elements in 1D)
#     x_coords = c_h.function_space.tabulate_dof_coordinates()[:, 0]
#     for _ in range(num_timesteps):
#         t += dt

#         assemble_transport_RHS()
#         solver.solve(b, c_h.x.petsc_vec)
#         c_h.x.scatter_forward()

#         c_.x.array[:] = c_h.x.array.copy()

#         print(f"Timestep t = {t:.2f}")
#         print("Max conc: ", mesh.comm.allreduce(c_h.x.array.max(), op=MPI.MAX))
#         print("Min conc: ", mesh.comm.allreduce(c_h.x.array.min(), op=MPI.MIN))

#         total_c = assemble_scalar(total_c_form)
#         total_c = mesh.comm.allreduce(total_c, op=MPI.SUM)
#         print(f"Total concentration: {total_c:.2e}")

#         snapshots.append(c_h.x.array.copy())
#         time_values.append(t)


#     # === Plot ===
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         raise RuntimeError("matplotlib is required to plot the solution.")

#     # plt.plot(c_h.function_space.tabulate_dof_coordinates()[:, 0], c_h.x.array)
#     # plt.xlabel("x")
#     # plt.ylabel("Concentration")
#     # plt.ylim(0, 1.1)
#     # plt.title("Concentration profile")
#     # plt.grid(True)
#     # plt.show()

#     import matplotlib.pyplot as plt
#     from matplotlib.animation import FuncAnimation

#     fig, ax = plt.subplots()
#     line, = ax.plot(x_coords, snapshots[0])
#     ax.set_ylim(min(map(np.min, snapshots)), max(map(np.max, snapshots))+0.1)
#     ax.set_xlabel("x")
#     ax.set_ylabel("Concentration")
#     ax.set_title("Concentration evolution")

#     def update(frame):
#         line.set_ydata(snapshots[frame])
#         ax.set_title(f"Time: {time_values[frame]:2.2f}")
#         return line,

#     ani = FuncAnimation(fig, update, frames=len(snapshots), interval=10, blit=False, repeat=False)


#     plt.show()