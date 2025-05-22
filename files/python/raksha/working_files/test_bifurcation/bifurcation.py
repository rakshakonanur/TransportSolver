from dolfinx.io import XDMFFile
from mpi4py import MPI

from ufl import dot, grad, jump, inner
from petsc4py import PETSc
from dolfinx.fem import FunctionSpace, dirichletbc, locate_dofs_geometrical, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx.mesh import locate_entities_boundary, meshtags, create_interval
from dolfinx.fem import Function, Constant, dirichletbc, locate_dofs_geometrical, functionspace, form, assemble_scalar
import dolfinx as dfx
import ufl
import numpy as np
import meshio
import vtk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

class Bifurcation:

    def __init__(self, u_val : float,
                       c_val,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        # self.L = L
        self.u_val = u_val
        self.D_value = 1e-3
        self.element_degree = 1
        self.write_output = write_output
        # # self.N = int((10* u_val * L) / (2 * self.D_value)) # Number of mesh cells: based on stability criterion of grid Pe
        # self.N = 10
        self.c_val = c_val
        # self.mesh = create_interval(MPI.COMM_WORLD, self.N, [0.0, L])
        self.read_mesh()
        self.mesh_tagging()

        # Temporal parameters
        self.T = 50
        self.dt = 0.5
        self.t = 0
        self.num_timesteps = int(self.T / self.dt)

    def read_mesh(self):
        """
        Read the mesh from a file and convert it to XDMF format.
        """
        # Read the mesh using meshio and write it to XDMF format
        # This is necessary because dolfinx does not support .msh files directly
        # Convert .msh to .vtu
        # mesh = meshio.read("bifurcation.msh")
        # meshio.write("before.vtu", mesh)


        # # Step 2: Apply vtkConnectivityFilter to keep the largest connected component
        # reader = vtk.vtkXMLUnstructuredGridReader()
        # reader.SetFileName("before.vtu")
        # reader.Update()
        # input_mesh = reader.GetOutput()

        # connectivity_filter = vtk.vtkConnectivityFilter()
        # connectivity_filter.SetInputData(input_mesh)
        # connectivity_filter.SetExtractionModeToLargestRegion()  # Keep the largest component
        # connectivity_filter.Update()

        # # Step 3: Write the filtered mesh to a new .vtu file

        # writer = vtk.vtkXMLUnstructuredGridWriter()
        # writer.SetFileName("after.vtu")
        # writer.SetInputData(connectivity_filter.GetOutput())
        # writer.Write()


        # # Step 4: Convert the filtered .vtu to .xdmf using meshio
        # filtered_mesh = meshio.read("after.vtu")
        # meshio.write("bifurcation.xdmf", filtered_mesh)

        # Read full Gmsh mesh
        msh = meshio.read("bifurcation.msh")

        # Extract line elements only
        line_cells = [cell for cell in msh.cells if cell.type == "line"]

        # Extract corresponding 'gmsh:physical' data for lines only
        line_cell_data = {}
        for key in msh.cell_data_dict:
            data_blocks = []
            for (ctype, data) in zip(msh.cells, msh.cell_data_dict[key]):
                if ctype == "line":
                    data_blocks.append(data)
            if data_blocks:
                line_cell_data[key] = data_blocks

        # Create a new mesh with only line elements
        line_mesh = meshio.Mesh(
            points=msh.points,
            cells=line_cells,
            cell_data=line_cell_data
        )

        # Write to XDMF
        line_mesh.write("bifurcation.xdmf")

        with XDMFFile(MPI.COMM_WORLD, "bifurcation.xdmf", "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")

        # Create connectivity between the mesh elements and their facets
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, 
                                            self.mesh.topology.dim - 1)



    def mesh_tagging(self):
        fdim = self.mesh.topology.dim - 1

        def left_boundary(x): return np.isclose(x[0], 0.0)
        def right_boundary(x): return np.isclose(x[0], 2.0)

        left_facets = locate_entities_boundary(self.mesh, fdim, left_boundary)
        right_facets = locate_entities_boundary(self.mesh, fdim, right_boundary)

        facet_indices = np.concatenate([left_facets, right_facets])
        facet_markers = np.concatenate([np.full(len(left_facets), 1, dtype=np.int32),
                                        np.full(len(right_facets), 2, dtype=np.int32)])
        self.facet_tag = meshtags(self.mesh, fdim, facet_indices, facet_markers)
        print("Facet indices: ", facet_indices, flush=True)
        print("Facet markers: ", facet_markers, flush=True)

    def compute_element_tangents(self):
        """Compute unit tangent vectors for each cell in a 1D mesh."""
        import numpy.linalg as la
        mesh = self.mesh
        dim = mesh.topology.dim

        # Get cell indices and connectivity
        cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
        conn = mesh.topology.connectivity(dim, 0)
        cell_nodes = [conn.links(i) for i in cells]

        # Access node coordinates
        coords = mesh.geometry.x

        tangents = []
        for nodes in cell_nodes:
            p0 = coords[nodes[0]]
            p1 = coords[nodes[1]]
            delta = p1 - p0
            tangent = delta / la.norm(delta)
            tangents.append(tangent)

        return cells, np.array(tangents)

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
        fdim = self.mesh.topology.dim - 1

        # Facet normal and integral measures
        n  = ufl.FacetNormal(self.mesh)
        # self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tag)
        dS = ufl.Measure("dS", domain=self.mesh)

        # === Function spaces ===
        Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=self.element_degree, shape=(self.mesh.geometry.dim,))
        V = functionspace(self.mesh, Pk_vec)
        u = Function(V)
        cells, self.tangents = self.compute_element_tangents()

        # Each dof in u lies in a cell, so we assign values per cell
        from dolfinx.fem.petsc import apply_lifting
        from dolfinx.fem import FunctionSpace
        from dolfinx.cpp.mesh import cell_entity_type

        # Loop over local cells and assign tangent vectors
        for i, cell in enumerate(cells):
            dofs = V.dofmap.cell_dofs(cell)
            for dof in dofs:
                u.x.array[dof*3:dof*3+3] = self.tangents[i] * self.u_val  # scale by desired speed

        # u.interpolate(lambda x: (1.0*x[0] - x[0] + self.u_val, 0.0*x[1], 0.0*x[2])) # div(u)=0 by construction
        # u.x.array[:] *= self.u_val  
        # u.x.array[:] = self.u_val  # constant velocity field
        print("Number of elements: ", self.mesh.topology.index_map(self.mesh.topology.dim).size_global, flush=True)

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

        # Surrounding concentration term:
        u_ex = lambda x: 1 + 0.0000001 *x[0] #x[0]**2 + 2*x[1]**2  
        x = ufl.SpatialCoordinate(self.mesh)
        # s = u_ex(x) # models the surrounding concentration
        s = Constant(self.mesh, dfx.default_scalar_type(1.0))
        r = Constant(self.mesh, dfx.default_scalar_type(0)) # for heat transfer, models the heat transfer coefficient
        g = dot(n, grad(u_ex(x))) # corresponding to the Neumann BC

        print("Total number of dofs: ", self.W.dofmap.index_map.size_global, flush=True)

        # === Boundary conditions ===
        def BoundaryConditionData(bc_type, marker, values):
            if bc_type == "Dirichlet":
                self.bc_func = Function(self.W)
                self.bc_func.x.array[:] = values
                dofs = self.facet_tag.find(marker)
                bc = dirichletbc(self.bc_func, dofs)

                # # For functions only
                # u_D = Function(self.W)
                # u_D.interpolate(values) 

                # dofs = self.facet_tag.find(marker)
                # bc = dirichletbc(u_D, dofs)
                return bc, "Dirichlet", None, None
            elif bc_type == "Neumann":
                L_neumann = inner(values, w) * ds(marker)
                return None, "Neumann", None, L_neumann
            elif bc_type == "Robin":
                r_val, s_val = values
                a_robin = r_val * inner(c, w) * ds(marker)
                L_robin = r_val * inner(s_val, w) * ds(marker)
                return None, "Robin", a_robin, L_robin
            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        # Define the Dirichlet and Robin conditions
        bcs_raw = [
            ("Dirichlet", 1, self.c_val[self.t]),
             ("Robin", 2, (r, s)),
            # ("Neumann",2, g)
        ]

        boundary_conditions = []
        bc_types = []
        robin_a_terms = []
        robin_L_terms = []

        for bc_type_name, marker, values in bcs_raw:
            bc, bctype, a_extra, L_extra = BoundaryConditionData(bc_type_name, marker, values)
            bc_types.append(bctype)
            if bc is not None:
                boundary_conditions.append(bc)
            if a_extra is not None:
                robin_a_terms.append(a_extra)
            if L_extra is not None:
                robin_L_terms.append(L_extra)

        # Variational forms
        a_time     = c * w / deltaT * ufl.dx
        a_advect   = dot(u, grad(c)) * w * ufl.dx
        a_diffuse  = dot(grad(c), grad(w)) * D * ufl.dx

        a = a_time + a_advect + a_diffuse + sum(robin_a_terms)
        L = (self.c_ / deltaT + f) * w * ufl.dx + sum(robin_L_terms)

        self.a_cpp = form(a)
        self.L_cpp = form(L)

        # Apply Dirichlet BCs
        self.bcs = boundary_conditions

        # Create output function in P1 space
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk))
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk_vec))
        
        # Interpolate it into the velocity function
        self.u_out.x.array[:] = u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c_form = form(self.c_h * ufl.dx)

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/bifurcation_conc_D=' + f'{D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.xdmf_c.write_function(self.c_out, self.t)

            out_str = './output/bifurcation_vel_D=' + f'{D.value}' + '.xdmf'
            self.xdmf_u = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_u.write_mesh(self.mesh)
            self.xdmf_u.write_function(self.u_out, self.t)

            # Write velocity to file
            vtx_u = dfx.io.VTXWriter(MPI.COMM_WORLD, './output/velocity.bp', [u], 'BP4')
            vtx_u.write(0)
            vtx_u.close()

        # === Linear system ===
        self.A = assemble_matrix(self.a_cpp, bcs=self.bcs)
        self.A.assemble()
        self.b = create_vector(self.L_cpp)

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

            val = self.c_val[_ - 1] if _ > 0 else self.c_val[0] # updates the inlet condition every time step
            self.bc_left = val
            self.bcs = [dirichletbc(self.bc_left, self.facet_tag.find(1), self.W)]  # Only apply at inle

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

            if self.write_output:
                # Write to file
                self.c_out.interpolate(self.c_h)
                self.xdmf_c.write_function(self.c_out, self.t)

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
        # Pe = self.u_val / self.D_value
        # c_true = (np.exp(Pe * self.x_coords) - np.exp(Pe * self.L)) / (1 - np.exp(Pe * self.L))
        # plt.plot(self.x_coords, c_true, label="Analytical solution")
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
    u_val = 0.1 # Velocity value
    k = 1 # Finite element polynomial degree

    # Create transport solver object
    transport_sim = Bifurcation(c_val=np.full(100, 1.0),
                                    # c_val=np.linspace(1, 1.5, 100),
                                    u_val=u_val,
                                    element_degree=k,
                                    write_output=write_output)
    transport_sim.setup()
    transport_sim.run()


# # Dirichlet BC for the inlet

        # self.bc_left_func = Function(self.W)
        # self.bc_left_func.x.array[:] = self.c_val[self.t] 

        # self.dof_left = locate_dofs_geometrical(self.W, lambda x: np.isclose(x[0], 0.0))
        # self.bcs = [dirichletbc(self.bc_left_func, self.dof_left)]

        # Robin BC for the outlet
        # self.bc_right = r * inner(u-s, w)* ds(2)

        # # # For constant boundary condition
        # self.bc_left = Constant(self.mesh, dfx.default_scalar_type(self.c_val[0]))
        # self.dof_left = locate_dofs_geometrical(self.W, lambda x: np.isclose(x[0], 0.0))
        # # self.bcs = [dirichletbc(self.bc_left, self.dof_left, self.W)]  # Only apply at inlet
        # self.bcs = [dirichletbc(self.bc_left, self.dof_left, self.W)]  # Only apply at inlet


        # === Variational Form ===
        # un = (dot(u, n) + abs(dot(u, n))) / 2.0