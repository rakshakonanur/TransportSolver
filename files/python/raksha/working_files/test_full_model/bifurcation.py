from dolfinx.io import XDMFFile
from mpi4py import MPI

from ufl import dot, grad, jump, inner, div, avg
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

        # Temporal parameters
        self.T = 2.5
        self.dt = 0.01
        self.t = 0
        self.num_timesteps = int(self.T / self.dt) + 1 # +1 to include the initial condition at t=0
        
        # self.mesh = create_interval(MPI.COMM_WORLD, self.N, [0.0, L])
        self.convert_mesh()
        if user_velocity:
            self.load_vtp()
            self.centerlineVelocity()
            self.import_velocity()
            self.results_to_vtk() # save the results to a VTK file
        else:
            self.u_val = u_val
        
        self.mesh_tagging()

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
    
    def centerlineVelocity(self, points_per_section=20):
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
        self.pressure = []  # initialize as a list

        from scipy.spatial import cKDTree

        for i in range(len(self.data_series)):
            print(f"Processing timestep {i}")

            self.area = self.data_series[i]["point_data"]["Area"]  # (N,)
            flowrate_1d = self.data_series[i]["point_data"]["Flowrate"]  # (N,)
            reynolds_1d = self.data_series[i]["point_data"]["Reynolds"]  # (N,)
            pressure_1d = self.data_series[i]["point_data"]["Pressure_mmHg"]  # (N,)
            self.mesh_3d_coords = self.data_series[i]["coords"]  # (N, 3)
            
            velocity_1d = flowrate_1d/self.area # calculate velocity from flowrate and area
            centerline = velocity_1d[::points_per_section] # save only one point for each cross-section
            centerlineFlowrate = flowrate_1d[::points_per_section] # save only one point for each cross-section
            self.centerlineFlow.append(centerlineFlowrate) # save the flowrate values for each timestep in each row

            self.centerlineVel.append(centerline)# save the velocity values for each timestep in each row
            self.pressure.append(pressure_1d[::points_per_section]) # save the pressure values for each timestep in each row
            self.centerlineCoords = self.mesh_3d_coords.reshape(-1, points_per_section, 3).mean(axis=1)
            
            output = pv.PolyData(self.centerlineCoords)

            output["Velocity_Magnitude"] = centerline
            output["Pressure_mmHg"] = pressure_1d[::points_per_section]
            output.save(f"output/averagedVelocity_{i:04d}.vtp")


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
        import pickle

        with open("data_series.pkl", "rb") as f:
            self.data_series = pickle.load(f)

        path = self.input_directory
        output_path = "output"
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

        # Commenting out nearest neighbor search
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

        # from scipy.interpolate import griddata

        # for i in range(num_timesteps):
        #     time = i * dt
        #     vtp_file = os.path.join(output_path + f"/averagedVelocity_{i:04d}.vtp")
        #     print(f"Processing timestep {i}: {vtp_file}", flush=True)

        #     # --- Load VTP and extract data ---
        #     vtp = pv.read(vtp_file)
        #     vtp_points = vtp.points
        #     scalar_velocity = vtp.point_data["Velocity_Magnitude"]

        #     # Linear interpolation (can use 'cubic' or 'nearest' as well)
        #     interpolated = griddata(
        #         vtp_points, scalar_velocity, dof_coords, method='linear', fill_value=0.0 # sets values that cannot be interpolated to 0.0
        #     )

        #     # If you want to avoid NaNs (outside convex hull), you can fall back to nearest for those points:
        #     nan_mask = np.isnan(interpolated)
        #     if np.any(nan_mask):
        #         interpolated[nan_mask] = griddata(
        #             vtp_points, scalar_velocity, dof_coords[nan_mask], method='nearest'
        #         )

        #     u_val[i, :] = interpolated

        self.u_val = u_val

        # Incorporating temporal interpolation
        print("Timestep interpolation: ", self.num_timesteps, flush=True)
        print("Number of timesteps in 1D NS solution: ", num_timesteps, flush=True)
        T = 250
        if self.num_timesteps + 1 != num_timesteps:
            from scipy.interpolate import interp1d
            # Suppose u_val has shape (num_timesteps, num_dofs)
            old_times = np.linspace(0, T, num=num_timesteps) # from the 1D NS solution
            new_times = np.linspace(0, self.T, num=self.num_timesteps) # interpolating to 1D advection-diffusion solution

            # Interpolate each dof across time
            f_interp = interp1d(old_times, u_val, axis=0, kind='linear', fill_value="extrapolate")
            u_val_interp = f_interp(new_times)  # shape: (self.num_timesteps, num_dofs)
            print("Interpolated velocity field from {} to {} timesteps.".format(num_timesteps, self.num_timesteps))
            self.u_val = u_val_interp

    def mesh_tagging(self):
        fdim = self.mesh.topology.dim - 1
        # Create connectivity between the mesh elements and their facets
        # self.mesh.topology.create_connectivity(self.mesh.topology.dim, 
                                            # self.mesh.topology.dim - 1)

        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        boundary_facets_indices = exterior_facet_indices(self.mesh.topology)
        inlet = np.array([0,0.41,0.34]) #updated with units
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

    from dolfinx.fem import FunctionSpace, Function
    from basix.ufl import element
    from scipy.spatial import cKDTree
    import numpy as np

    # def assign_velocity_field(self): # NOT WORKING
    #     """
    #     Interpolate scalar velocity × tangent vector field using u.interpolate().
    #     Works with higher-order elements robustly.
    #     """
    #     # 1. Get coordinates of mesh dofs (not just P1!)
    #     V_scalar = functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), self.element_degree))
    #     dof_coords = V_scalar.tabulate_dof_coordinates()

    #     # 2. Build KDTree from cell centers (from compute_element_tangents)
    #     dim = self.mesh.topology.dim
    #     conn = self.mesh.topology.connectivity(dim, 0)
    #     cell_nodes = [conn.links(i) for i in range(len(self.tangents))]
    #     tangent_points = np.array([self.mesh.geometry.x[nodes] for nodes in cell_nodes])
    #     tangent_centers = tangent_points.mean(axis=1)  # shape: (num_cells, 3)
    #     tree = cKDTree(tangent_centers)

    #     # 3. Store scalar velocity as an array
    #     scalar_vals = self.u_val[self.t, :]  # shape: (num_dofs,)

    #     # 4. Build a callable for interpolation
    #     tangents = self.tangents

    #     def velocity_callable(x):
    #         """
    #         Interpolation of u(x) = scalar_val * tangent using nearest cell center.
    #         x: shape (3, N)
    #         return shape: (3, N)
    #         """
    #         # Interpolate scalar velocity from known dof coordinates using nearest-neighbor
    #         _, dof_indices = tree.query(x.T)  # nearest tangent per point
    #         local_tangents = tangents[dof_indices]  # shape: (N, 3)

    #         # Nearest scalar val per point (assumes x close to known dof coords)
    #         _, nearest_dofs = tree.query(x.T)
    #         scalar_interp = scalar_vals[nearest_dofs]  # shape: (N,)

    #         # Multiply scalar × tangent per point
    #         return (scalar_interp[:, np.newaxis] * local_tangents).T  # shape (3, N)

    #     # 5. Interpolate into vector field
    #     self.u.interpolate(velocity_callable)                   

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
        self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tag)
        self.dS = ufl.Measure("dS", domain=self.mesh)

        # === Function spaces ===
        self.Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
        V = functionspace(self.mesh, self.Pk_vec)
        self.u = Function(V)
        self.cells, self.tangents = self.compute_element_tangents()

        # self.assign_velocity_field()

        # Each dof in u lies in a cell, so we assign values per cell
        from dolfinx.fem.petsc import apply_lifting
        from dolfinx.fem import FunctionSpace
        from dolfinx.cpp.mesh import cell_entity_type

        # Loop over local cells and assign tangent vectors --> MIGHT NEED TO BE CHANGED
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
        self.r = Constant(self.mesh, dfx.default_scalar_type(1.0)) # for heat transfer, models the heat transfer coefficient
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
        self.P1 = element("Lagrange", self.mesh.basix_cell(), degree=1) # Linear Lagrange elements
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, self.P1))
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, self.Pk_vec))
        self.robin_error_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, self.Pk))
        
        # Interpolate it into the velocity function
        self.u_out.x.array[:] = self.u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c_form = form(self.c_h * self.dx)

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/bifurcation_conc_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.c_out.interpolate(self.c_h)  # Interpolate the concentration function
            self.xdmf_c.write_function(self.c_out, self.t)

            out_str = './output/bifurcation_vel_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_u = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_u.write_mesh(self.mesh)
            self.xdmf_u.write_function(self.u_out, self.t)

            # out_str = './output/boundary_conditions_D=' + f'{self.D.value}' + '.xdmf'
            # self.xdmf_er = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            # self.xdmf_er.write_mesh(self.mesh)
            # self.xdmf_er.write_function(self.robin_error_out, self.t)

            # Write velocity to file
            vtx_u = dfx.io.VTXWriter(MPI.COMM_WORLD, './output/velocity.bp', [self.u], 'BP4')
            vtx_u.write(0)
            vtx_u.close()

        # self.assemble_linear_system()

        
    def assemble_transport_LHS(self):
        """ Assemble the linear system. """
        # Variational forms
        a_time     = self.c * self.w / self.deltaT * self.dx
        a_advect   = dot(self.u, grad(self.c)) * self.w * self.dx
        a_diffuse  = dot(grad(self.c), grad(self.w)) * self.D * self.dx

        a = a_time + a_advect + a_diffuse + sum(self.robin_a_terms)
        L = (self.c_ / self.deltaT + self.f) * self.w * self.dx + sum(self.robin_L_terms)

        # # Incorporate the upwind velocity term
        # b_mag = ufl.sqrt(ufl.dot(self.u, self.u)) + 1e-10
        # tau = h / (2 * b_mag)  # standard SUPG τ

        # # --- Strong residual ---
        # residual = -ufl.div(D * ufl.grad(c)) + ufl.dot(self.u, ufl.grad(c)) - f

        # # SUPG terms
        # a += tau * ufl.dot(self.u, ufl.grad(w)) * residual * ufl.dx
        # L += tau * ufl.dot(self.u, ufl.grad(w)) * f * ufl.dx

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

    def check_residuals(self, visualize=False):

        """
        Compute and plot the residuals of the steady-state advection-diffusion equation.
        """
        # === Compute residual norm ===
        # Step 1: Copy RHS vector
        residual_vec = self.b.copy()

        # Step 2: Compute A * c_h
        A_ch = self.b.copy()  # Temporary PETSc vector
        self.A.mult(self.c_h.x.petsc_vec, A_ch)

        # Step 3: Subtract A * c_h from b
        residual_vec.axpy(-1.0, A_ch)  # residual_vec = b - A_ch

        # Step 4: Compute residual norm (e.g. L2 norm)
        res_norm = residual_vec.norm()
        print(f"Residual norm ||r|| = {res_norm:.2e}")
        self.total_residual = res_norm**2  # Store the squared norm for later use

    def check_boundary_conditions(self, visualize=False):

        """
        Compute and plot the error in (Robin) BC implementation on the outlet.
        Evaluates D * grad(c) · n + r * c - r * s over the outlet boundary.
        Should be near zero if the Robin BC is implemented correctly.
        """

        W = self.W
        D = self.D
        r = self.r
        s = self.s
        n = self.n

        # Create function for error evaluation
        from dolfinx.fem import assemble_scalar, form
        import ufl
        # Compute the integrand over outlet facets (marker=2)
        normal_flux = D * dot(grad(self.c_h), n)
        robin_expr = normal_flux - (r * self.c_h - r * s)

        # Square it to compute L2 error
        error_form = robin_expr**2 * self.ds(2) # calculates square of L2 error over the outlet facets
        error_squared = assemble_scalar(form(error_form)) # assemble into a scalar, by converting symbolic UFL form to Fenicsx
        self.total_error = self.mesh.comm.allreduce(error_squared, op=MPI.SUM) # gather all the errors from all processes in case of parallel execution  

    def run(self):
        """ Run transport simulations. """

        # Allocate list to store time series snapshots
        self.snapshots = []
        self.error = []
        self.residuals = []
        self.time_values = []
      
        for _ in range(self.num_timesteps - 1):
            
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
                    scalar_val = self.u_val[_, dof]
                    self.u.x.array[dof*3:dof*3+3] = self.tangents[i] * scalar_val
            # self.assign_velocity_field()
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

                self.u_out.interpolate(self.u)
                self.xdmf_u.write_function(self.u_out, self.t)

                self.check_boundary_conditions(visualize=True)
                self.check_residuals(visualize=True)
                print(f"L2 error of Robin BC at outlet: {np.sqrt(self.total_error):.4e}") # find the square root
                print(f"L2 error of residuals: {np.sqrt(self.total_residual):.4e}") # find the square root

            self.snapshots.append(self.c_h.x.array.copy())
            self.error.append(np.sqrt(self.total_error))  # Store the error for each timestep
            self.residuals.append(np.sqrt(self.total_residual))
            self.time_values.append(self.t)

        fig, ax = plt.subplots()
        x_data_err, y_data_err = [], []
        x_data_res, y_data_res = [], []

        line_err, = ax.plot([], [], label="Robin BC Error", color='blue')
        line_res, = ax.plot([], [], label="Residual", color='red')

        ax.set_xlim(self.time_values[0], self.time_values[-1])
        y_min = min(min(self.error), min(self.residuals)) - 0.01
        y_max = max(max(self.error), max(self.residuals)) + 0.01
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Error Evolution")
        ax.legend()

        def update(frame):
            # Append new data points
            x_data_err.append(self.time_values[frame])
            y_data_err.append(self.error[frame])

            x_data_res.append(self.time_values[frame])
            y_data_res.append(self.residuals[frame])

            # Update both lines
            line_err.set_data(x_data_err, y_data_err)
            line_res.set_data(x_data_res, y_data_res)

            ax.set_title(f"Time: {self.time_values[frame]:.2f}")
            return line_err #, line_res

        ani = FuncAnimation(fig, update, frames=len(self.error), interval=100, blit=False, repeat=False)
        plt.show()


        # fig, ax = plt.subplots()
        # x_data, y_data = [], []
        # line, = ax.plot([], [], label="Error over time")

        # ax.set_xlim(self.time_values[0], self.time_values[-1])
        # ax.set_ylim(min(self.error) - 0.01, max(self.error) + 0.01)
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Error")
        # ax.set_title("Error evolution")

        # def update(frame):
        #     x_data.append(self.time_values[frame])
        #     y_data.append(self.error[frame])
        #     line.set_data(x_data, y_data)
        #     ax.set_title(f"Time: {self.time_values[frame]:.2f}")
        #     return line,

        # ani = FuncAnimation(fig, update, frames=len(self.error), interval=100, blit=False, repeat=False)

        
        # fig, ax = plt.subplots()
        # x_data, y_data = [], []
        # line, = ax.plot([], [], label="Error over time")

        # ax.set_xlim(self.time_values[0], self.time_values[-1])
        # ax.set_ylim(min(self.residuals) - 0.01, max(self.residuals) + 0.01)
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Residuals")
        # ax.set_title("Residuals evolution")

        # def update_residual(frame):
        #     x_data.append(self.time_values[frame])
        #     y_data.append(self.residuals[frame])
        #     line.set_data(x_data, y_data)
        #     ax.set_title(f"Time: {self.time_values[frame]:.2f}")
        #     return line,

        # plt.legend()
        # plt.show()
        # ani.save("error_scalar_evolution.mp4", writer="ffmpeg", fps=10)
            
        return self.snapshots, self.time_values
    
    def results_to_vtk(self, output_file="results.vtk"):
        """
        Save the results to a VTK file.
        Interpolates the concentration and velocity fields from xdmf files to 3D VTK format.
        Args:
            output_file (str): The name of the output VTK file.
        """
        from scipy.interpolate import RBFInterpolator
        dim = self.mesh.topology.dim

        # --- Centerline Coordinates and Area ---
        self.mesh_3d_coords = self.data_series[0]["coords"]
        self.centerlineCoords = self.mesh_3d_coords.reshape(-1, 20, 3).mean(axis=1)   # (N, 3)
        centerlineCoords = self.centerlineCoords
        area = self.data_series[0]["point_data"]["Area"][::20]                        # (N,)

        # --- Mesh Points and Connectivity ---
        mesh_points = self.mesh.geometry.x                                            # (M, 3)

        # Get cell-to-vertex connectivity using topology
        self.mesh.topology.create_connectivity(dim, 0)
        conn = self.mesh.topology.connectivity(dim, 0)
        num_cells = self.mesh.topology.index_map(dim).size_local

        # Reconstruct cell list as (num_cells, 2)
        mesh_cells = np.vstack([
            conn.array[conn.offsets[i]:conn.offsets[i+1]]
            for i in range(num_cells)
        ])

        # --- Step 1: RBF Interpolation ---
        rbf_interp = RBFInterpolator(
            centerlineCoords,
            area,
            kernel='linear',     # Try 'thin_plate_spline' for smoother results
            neighbors=10
        )
        interpolated_area = rbf_interp(mesh_points)  # (M,)

        # --- Step 2: Prepare VTK lines ---
        vtk_lines = np.hstack([
            np.full((mesh_cells.shape[0], 1), 2, dtype=np.int32),
            mesh_cells
        ]).flatten()

        # --- Step 3: Create PyVista Line Mesh ---
        line_mesh = pv.PolyData()
        line_mesh.points = mesh_points
        line_mesh.lines = vtk_lines
        line_mesh['InterpolatedArea'] = interpolated_area

        # --- Step 4: Save ---
        line_mesh.save(output_file)
        print(f"✅ Exported to '{output_file}'") 

        # # Load the concentration field from snapshots
        # concentration_centerline = np.zeros((self.num_timesteps, mesh_points.shape[0]))
        # for t in range(self.num_timesteps):
        #     concentration_centerline[t, :] = self.snapshots[t]

        radius = np.sqrt(interpolated_area / np.pi)                      # shape (N,)
        from collections import OrderedDict

        # 1. Read the XDMF file using meshio
        mesh = meshio.read("bifurcation.xdmf")

        # print("Cell data keys:", mesh.cell_data_dict.keys())
        # print("Point data keys:", mesh.point_data.keys())
        # print("Cell data contents:")
        # for key, data_dict in mesh.cell_data_dict.items():
        #     print(f"  Cell type: {key}")
        #     for name, array in data_dict.items():
        #         print(f"    {name} -> shape: {array.shape} dtype: {array.dtype} unique values: {np.unique(array)}")
        # print("Point data contents:")
        # for name, array in mesh.point_data.items():
        #     print(f"  {name} -> shape: {array.shape} dtype: {array.dtype} unique values: {np.unique(array)}")
    

        # 2. Identify the line cells and physical tags
        line_cells = None
        for cell_block in mesh.cells:
            if cell_block.type == "line":
                line_cells = cell_block.data
                break

        if line_cells is None:
            raise RuntimeError("No line cells found.")

        physical_tags = None
        if "gmsh:physical" in mesh.cell_data_dict:
            physical_tags_dict = mesh.cell_data_dict["gmsh:physical"]
            if "line" in physical_tags_dict:
                physical_tags = physical_tags_dict["line"]

        if physical_tags is None:
            raise RuntimeError("No physical tags found for line cells.")

        print(f"Found physical tags for {len(physical_tags)} lines")


        # 3. Group lines by branch ID (physical tag)
        branches = {}
        for i, tag in enumerate(physical_tags):
            branches.setdefault(tag, []).append(line_cells[i])

        # 4. Function to create tube mesh for each branch
        def create_branch_tube(branch_lines, points, radius):
            unique_pts = list(OrderedDict.fromkeys([pt for line in branch_lines for pt in line]))
            branch_points = points[unique_pts]
            # Ensure radius is a NumPy array
            radius = np.asarray(radius)
            branch_radii = radius[unique_pts]
            line = pv.lines_from_points(branch_points, close=False)
            line.point_data["Radius"] = branch_radii
            
            tube = line.tube(scalars="Radius", absolute=True)
            return tube

        # 5. Generate tubes and merge into a single mesh
        points = mesh.points
        combined_mesh = None
        for branch_id, branch_lines in branches.items():
            print(f"Processing branch {branch_id} with {len(branch_lines)} lines")
            tube = create_branch_tube(branch_lines, points, radius)
            if combined_mesh is None:
                combined_mesh = tube
            else:
                combined_mesh = combined_mesh.merge(tube)

        # 6. Save combined mesh to one VTP file
        combined_mesh.save("all_branches.vtp")
        print("Saved all branches combined to 'all_branches.vtp'")

# === Time loop ===
if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True
    user_velocity = True
    L = 1.0
    u_val = 0.5 # Velocity value
    k = 1 # Finite element polynomial degree
    path="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output/1D_Output/071725/Run5_25branches"
    # Create transport solver object
    transport_sim = Bifurcation(c_val=np.full(251, 5.0),
                                    # c_val= np.concatenate([np.linspace(0, 2, 75), np.linspace(2, 0, 75), np.linspace(0, 0, 100)]),
                                    user_velocity=user_velocity,
                                    u_val=u_val,
                                    element_degree=k,
                                    write_output=write_output,
                                    input_directory=path)
    transport_sim.setup()
    transport_sim.run()

