from dolfinx.io import XDMFFile
from mpi4py import MPI

from ufl import dot, grad, jump, inner
from petsc4py import PETSc
from dolfinx.fem import FunctionSpace, dirichletbc, locate_dofs_geometrical, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities, locate_entities_boundary, meshtags
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx.mesh import locate_entities_boundary, meshtags, create_interval, exterior_facet_indices
from dolfinx.fem import Function, Constant, dirichletbc, locate_dofs_geometrical, functionspace, form, assemble_scalar
import dolfinx as dfx
import ufl
import numpy as np
import meshio
import vtk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import meshio
import pyvista as pv
import os
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy
import glob
import re

class Bifurcation:

    def __init__(self, user_velocity: bool,
                       u_val : float,
                       c_val,
                       input_directory: str,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        # self.L = L
        self.D_value = 1e-3
        self.element_degree = 1
        self.write_output = write_output
        # # self.N = int((10* u_val * L) / (2 * self.D_value)) # Number of mesh cells: based on stability criterion of grid Pe
        # self.N = 10
        self.c_val = c_val
        self.input_directory = input_directory
        
        # self.mesh = create_interval(MPI.COMM_WORLD, self.N, [0.0, L])
        self.convert_mesh()
        if user_velocity:
            # self.load_vtp()
            self.centerlineVelocity()
            self.import_velocity()
        else:
            self.u_val = u_val
        
        self.mesh_tagging()

        # Temporal parameters
        self.T = 100
        self.dt = 1
        self.t = 0
        self.num_timesteps = int(self.T / self.dt)

    def convert_mesh(self):
        """
        Read the mesh from a file and convert it to XDMF format.
        """
       
        # Read full Gmsh mesh
        msh = meshio.read("branched_network.msh")
        # Extract only "line" elements
        line_cells = [cell for cell in msh.cells if cell.type == "line"]
        if not line_cells:
            raise RuntimeError("No line cells found in the mesh.")
        # meshio expects a list of (cell_type, numpy_array)
        line_cells = [("line", line_cells[0].data)]
        # Optionally, also filter cell data for lines only
        line_cell_data = {}
        if "gmsh:physical" in msh.cell_data_dict:
            # meshio expects a dict of {cell_type: [data_array]}
            line_cell_data = {"gmsh:physical": [msh.cell_data_dict["gmsh:physical"]["line"]]}
        # Create a new mesh with only line elements
        line_mesh = meshio.Mesh(
            points=msh.points,
            cells=line_cells,
            cell_data=line_cell_data if line_cell_data else None
        )
        # Write to XDMF
        line_mesh.write("bifurcation.xdmf")

        # for cell in msh.cells:
        #     line_cells = [cell for cell in msh.cells if cell.type == "line"]
        # for key in msh.cell_data_dict["gmsh:physical"].keys():
        #     data_blocks = []
        #     if key == "line":
        #         line_cell_data = msh.cell_data_dict["gmsh:physical"][key]   
        # print("Line cells: ", line_cells, flush=True)
        # print("Line cell data: ", line_cell_data, flush=True)
        # line_mesh = meshio.Mesh(points = msh.points,
        #                    cells = line_cells,
        #                    cell_data= line_cell_data)
        # line_mesh.write("bifurcation.xdmf")
        # meshio.write(output_directory + ".xdmf", msh)

        # msh = meshio.read("branched_network.msh")

        # # Extract only "line" elements (not "polyline" or others)
        # line_cells = [cell for cell in msh.cells if cell.type == "line"]
        # print("Line cells: ", line_cells, flush=True)

        # # Optionally, also filter cell data for lines only
        # line_cell_data = {}
        # if "gmsh:physical" in msh.cell_data_dict:
        #     for key, data in msh.cell_data_dict["gmsh:physical"].items():
        #         if key == "line":
        #             line_cell_data[key] = data

        # print("Line cell data: ", line_cell_data, flush=True)

        # # Create a new mesh with only line elements
        # line_mesh = meshio.Mesh(
        #     points=msh.points,
        #     cells=line_cells,
        #     cell_data=line_cell_data  # Uncomment if you want to include cell data
        # )

        # # Write to XDMF
        # line_mesh.write("bifurcation.xdmf")

        with XDMFFile(MPI.COMM_WORLD, "bifurcation.xdmf", "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")

        # Create connectivity between the mesh elements and their facets
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, 
                                            self.mesh.topology.dim - 1)
    
    def load_vtp(self):
        """
        Load VTP time-step files from the specified directory and extract data.
        Need time-dependent velocity data for the advection-diffusion equation.
        The function reads all VTP files in the specified directory, extracts point coordinates,
        point data, cell data, and field data, and stores them in a list of dictionaries.
        Each dictionary corresponds to a time step and contains the filename, coordinates,
        point data, cell data, and field data.  

        """

        directory = self.input_directory

        def extract_arrays(data_obj):
            return {
                data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                for i in range(data_obj.GetNumberOfArrays())
            }

        # Find all model files in the specified directory
        print("Searching for model files in directory:", directory)
        
        # Get a sorted list of all VTP files
        vtp_files = sorted(glob.glob(os.path.join(directory, "*.vtp")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
        all_data = []

        for file in vtp_files:
            print(f"Reading: {file}")
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(file)
            reader.Update()
            polydata = reader.GetOutput()

            def extract_arrays(data_obj):
                return {
                    data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                    for i in range(data_obj.GetNumberOfArrays())
                }
            
            # Extract point coordinates
            points = polydata.GetPoints()

            # Convert to NumPy array
            num_points = points.GetNumberOfPoints()

            timestep_data = {
                "filename": file,
                "coords": np.array([points.GetPoint(i) for i in range(num_points)]),
                "point_data": extract_arrays(polydata.GetPointData()),
                "cell_data": extract_arrays(polydata.GetCellData()),
                "field_data": extract_arrays(polydata.GetFieldData())
            }

            all_data.append(timestep_data)
        import pickle

        with open("data_series.pkl", "wb") as f:
            pickle.dump(all_data, f)
        self.data_series = all_data
    
    def centerlineVelocity(self):
        """
        Interpolate the velocity data from the VTP files onto the centerline of the reference model.
        Args:
            data_series (list): List of dictionaries containing the VTP data.
            ref_model (pyvista.PolyData): Reference model for interpolation.
        """
        import pickle
        with open("data_series.pkl", "rb") as f:
            self.data_series = pickle.load(f)
        
        print("Number of timesteps:", len(self.data_series))
        # Iterate over each timestep in the data series
        self.centerlineVel = []  # initialize as a list
        self.centerlineFlow = []  # initialize as a list

        for i in range(len(self.data_series)):
            print(f"Processing timestep {i}")

            area_1d = self.data_series[i]["point_data"]["Area"]  # (N,)
            flowrate_1d = self.data_series[i]["point_data"]["Flowrate"]  # (N,)
            reynolds_1d = self.data_series[i]["point_data"]["Reynolds"]  # (N,)
            coords_1d = self.data_series[i]["coords"]  # (N, 3)
            
            velocity_1d = flowrate_1d/area_1d # calculate velocity from flowrate and area
            centerline = velocity_1d[::20] # save only one point for each cross-section
            centerlineFlowrate = flowrate_1d[::20] # save only one point for each cross-section
            self.centerlineFlow.append(centerlineFlowrate) # save the flowrate values for each timestep in each row
            self.centerlineVel.append(centerline)# save the velocity values for each timestep in each row
            centerlineCoords = coords_1d.reshape(-1, 20, 3).mean(axis=1)
            
            output = pv.PolyData(centerlineCoords)

            output["Velocity_Magnitude"] = centerline
            output.save(f"output/averagedVelocity_{i:04d}.vtp")

        self.centerlineVelocity = np.array(self.centerlineVelocity)
        print("✅ Projected flowrate values saved to 'averageVelocity.vtp'")

    def import_velocity(self):
        
        """
        Interpolates scalar velocity from VTP centerlines onto a 1D mesh and writes XDMF time series using DOLFINx.
        """
        import dolfinx
        import numpy as np
        import pyvista as pv
        from dolfinx.fem import Function, functionspace
        from dolfinx.io import XDMFFile
        from basix.ufl import element
        from mpi4py import MPI
        from scipy.spatial import cKDTree
        import meshio

        path = self.input_directory
        output_path = "output/"
        num_timesteps = len(self.data_series)
        dt = 1  # Adjust as needed

        # === Load DOLFINx-compatible mesh ===
        with XDMFFile(MPI.COMM_WORLD, "bifurcation.xdmf", "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")

        # Create scalar P1 function space
        V = functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1))
        # u_val = Function(V)

        # === Get coordinates of mesh dofs ===
        dof_coords = V.tabulate_dof_coordinates()
        num_dofs = dof_coords.shape[0]

        # Allocate u_val as a 2D array for all timesteps and dofs
        u_val = np.zeros((num_timesteps, num_dofs))

        # Create connectivity between the mesh elements and their facets
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, 
                                            self.mesh.topology.dim - 1)

        tree = None  # KDTree for nearest neighbor search
        for i in range(num_timesteps):
            time = i * dt
            vtp_file = os.path.join(output_path + f"/averagedVelocity_{i:04d}.vtp")
            print(f"Processing timestep {i}: {vtp_file}", flush=True)

            # --- Load VTP and extract data ---
            vtp = pv.read(vtp_file)
            vtp_points = vtp.points
            scalar_velocity = vtp.point_data["Velocity_Magnitude"]

            if tree is None:
                tree = cKDTree(vtp_points)

            # Nearest neighbor interpolation
            distances, indices = tree.query(dof_coords)
            interpolated = scalar_velocity[indices]
            # interpolated = np.nan_to_num(interpolated)  # Avoid NaNs

            u_val[i,:] = interpolated

        self.u_val = u_val

    def mesh_tagging(self):
        fdim = self.mesh.topology.dim - 1
        # Create connectivity between the mesh elements and their facets
        # self.mesh.topology.create_connectivity(self.mesh.topology.dim, 
                                            # self.mesh.topology.dim - 1)

        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        boundary_facets_indices = exterior_facet_indices(self.mesh.topology)
        inlet = np.array([3.0, 3.05, 3.4])
        # print("Boundary facets indices: ", boundary_facets_indices, flush=True)
        # def left_boundary(x): return np.isclose(x[0], 0.0)
        # def right_boundary(x): return np.isclose(x[0], 2.0)
        tol = 1e-6

        def near_inlet(x):
            return np.isclose(x[0],inlet[0]) & np.isclose(x[1],inlet[1]) & np.isclose(x[2],inlet[2])

        inlet_facets = locate_entities_boundary(self.mesh, fdim, near_inlet)

        # Extract outlet facets: boundary facets excluding inlet facets
        outlet_facets = np.setdiff1d(boundary_facets_indices, inlet_facets)

        # inlet_facets = [boundary_facets_indices[0]] # first element corresponds to inlet
        # outlet_facets = boundary_facets_indices[1:] # all other elements correspond to outlets

        facet_indices = np.concatenate([inlet_facets, outlet_facets])
        facet_markers = np.concatenate([np.full(len(inlet_facets), 1, dtype=np.int32),
                                        np.full(len(outlet_facets), 2, dtype=np.int32)])
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
        self.n  = ufl.FacetNormal(self.mesh)
        # self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tag)
        self.dS = ufl.Measure("dS", domain=self.mesh)

        # === Function spaces ===
        self.Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=self.element_degree, shape=(self.mesh.geometry.dim,))
        V = functionspace(self.mesh, self.Pk_vec)
        self.u = Function(V)
        self.cells, self.tangents = self.compute_element_tangents()

        # Each dof in u lies in a cell, so we assign values per cell
        from dolfinx.fem.petsc import apply_lifting
        from dolfinx.fem import FunctionSpace
        from dolfinx.cpp.mesh import cell_entity_type

        # Loop over local cells and assign tangent vectors
        for i, cell in enumerate(self.cells):
            dofs = V.dofmap.cell_dofs(cell)
            for dof in dofs:
                scalar_val = self.u_val[self.t, dof]
                self.u.x.array[dof*3:dof*3+3] = self.tangents[i] * scalar_val
        self.u.x.scatter_forward()
        # u.interpolate(lambda x: (1.0*x[0] - x[0] - self.u_val, 0.0*x[1], 0.0*x[2])) # div(u)=0 by construction
        # u.x.array[:] *= self.u_val  
        # u.x.array[:] = self.u_val  # constant velocity field
        print("Number of elements: ", self.mesh.topology.index_map(self.mesh.topology.dim).size_global, flush=True)

        self.Pk = element("Lagrange", self.mesh.basix_cell(), degree=self.element_degree)
        self.W = functionspace(self.mesh, self.Pk)
        print("Total number of concentration dofs: ", self.W.dofmap.index_map.size_global, flush=True)
        
        
        # === Trial, test, and solution functions ===
        self.c, self.w = ufl.TrialFunction(self.W), ufl.TestFunction(self.W)
        self.c_h = Function(self.W)
        self.c_ = Function(self.W)

        #------VARIATIONAL FORM------#
        self.D = Constant(self.mesh, dfx.default_scalar_type(self.D_value))
        self.f = Constant(self.mesh, dfx.default_scalar_type(0.0))
        self.h = Constant(self.mesh, dfx.default_scalar_type(1.0))
        self.deltaT = Constant(self.mesh, dfx.default_scalar_type(self.dt))
        self.beta = Constant(self.mesh, dfx.default_scalar_type(10.0))
        self.hf = ufl.CellDiameter(self.mesh)

        # Surrounding concentration term:
        self.u_ex = lambda x: 1 + 0.0000001 *x[0] #x[0]**2 + 2*x[1]**2  
        self.x = ufl.SpatialCoordinate(self.mesh)
        # s = u_ex(x) # models the surrounding concentration
        self.s = Constant(self.mesh, dfx.default_scalar_type(1.0))
        self.r = Constant(self.mesh, dfx.default_scalar_type(0)) # for heat transfer, models the heat transfer coefficient
        self.a = Constant(self.mesh, dfx.default_scalar_type(10))
        self.g = dot(self.n, grad(self.u_ex(self.x))) # corresponding to the Neumann BC

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
                L_neumann = inner(values, self.w) * self.ds(marker)
                return None, "Neumann", None, L_neumann
            elif bc_type == "Robin":
                r_val, a_val, s_val = values
                a_robin = a_val * inner(self.c, self.w) * self.ds(marker)
                L_robin = r_val * inner(s_val, self.w) * self.ds(marker)
                return None, "Robin", a_robin, L_robin
            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        # Define the Dirichlet and Robin conditions
        bcs_raw = [
            ("Dirichlet", 1, self.c_val[self.t]),
             ("Robin", 2, (self.r, self.r, self.s)),
            # ("Neumann",2, g)
        ]

        self.boundary_conditions = []
        bc_types = []
        self.robin_a_terms = []
        self.robin_L_terms = []

        for bc_type_name, marker, values in bcs_raw:
            bc, bctype, a_extra, L_extra = BoundaryConditionData(bc_type_name, marker, values)
            bc_types.append(bctype)
            if bc is not None:
                self.boundary_conditions.append(bc)
                print(f"Boundary condition: {bc_type_name} on facet {marker} with value {values}", flush=True)
            if a_extra is not None:
                self.robin_a_terms.append(a_extra)
            if L_extra is not None:
                self.robin_L_terms.append(L_extra)

        self.assemble_transport_LHS()

        # Apply Dirichlet BCs
        self.bcs = self.boundary_conditions

        # Create output function in P1 space
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, self.Pk))
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, self.Pk_vec))
        
        # Interpolate it into the velocity function
        self.u_out.x.array[:] = self.u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c_form = form(self.c_h * ufl.dx)

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/bifurcation_conc_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.xdmf_c.write_function(self.c_out, self.t)

            out_str = './output/bifurcation_vel_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_u = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_u.write_mesh(self.mesh)
            self.xdmf_u.write_function(self.u_out, self.t)

            # Write velocity to file
            vtx_u = dfx.io.VTXWriter(MPI.COMM_WORLD, './output/velocity.bp', [self.u], 'BP4')
            vtx_u.write(0)
            vtx_u.close()

        self.assemble_linear_system()


    def assemble_transport_LHS(self):
        """ Assemble the linear system. """
        # Variational forms
        a_time     = self.c * self.w / self.deltaT * ufl.dx
        a_advect   = dot(self.u, grad(self.c)) * self.w * ufl.dx
        a_diffuse  = dot(grad(self.c), grad(self.w)) * self.D * ufl.dx

        a = a_time + a_advect + a_diffuse + sum(self.robin_a_terms)
        L = (self.c_ / self.deltaT + self.f) * self.w * ufl.dx + sum(self.robin_L_terms)

        self.a_cpp = form(a)
        self.L_cpp = form(L)

    def assemble_linear_system(self):
        """ Assemble the linear system. """
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
            self.bcs = [dirichletbc(self.bc_left, self.facet_tag.find(1), self.W)]  # Only apply at inlet

            V = self.u.function_space
            for i, cell in enumerate(self.cells):
                dofs = V.dofmap.cell_dofs(cell)
                for dof in dofs:
                    scalar_val = self.u_val[self.t, dof]
                    self.u.x.array[dof*3:dof*3+3] = self.tangents[i] * scalar_val
            self.u.x.scatter_forward()
            self.assemble_transport_LHS()
            self.assemble_linear_system()
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
            
        return self.snapshots, self.time_values


# === Time loop ===
if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True
    user_velocity = True
    L = 1.0
    u_val = 0.5 # Velocity value
    k = 1 # Finite element polynomial degree
    path="/mnt/c/Users/rkona/Documents/syntheticVasculature/1D Output/052125/Run4_100branches"
    # Create transport solver object
    transport_sim = Bifurcation(c_val=np.full(101, 10.0),
                                    # c_val= np.concatenate([np.linspace(0, 2, 75), np.linspace(2, 0, 75), np.linspace(0, 0, 100)]),
                                    user_velocity=user_velocity,
                                    u_val=u_val,
                                    element_degree=k,
                                    write_output=write_output,
                                    input_directory=path)
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