import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad, inner
from sys               import argv
from mpi4py            import MPI
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx.fem import Constant
from dolfinx.mesh import locate_entities_boundary, meshtags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

""" Use a Galerkin method to
    solve a variational problem for the advection-diffusion equation

    dc/dt + div(J) = f

    with

    J = c*u - D*grad(c)

    where
        - c is solute concentration
        - u is velocity
        - D is diffusion coefficient
        - f is a source term
"""

print = PETSc.Sys.Print

# Generate mesh
if len(argv) < 2:
    raise RuntimeError("Specify number of mesh cells as command line argument.")
mesh = dfx.mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=int(argv[1]))
LEFT = 1; RIGHT = 2

# Simulation parameters
D_value = 1e-3
k = 1 # Element degree
t = 0 # Initial time
T = 10 # Final time
dt = 0.1 # Timestep size
num_timesteps = int(T / dt)
n = ufl.FacetNormal(mesh)
dx = ufl.Measure("dx", domain=mesh) # Cell integrals


def left_boundary(x): return np.isclose(x[0], 0.0)
def right_boundary(x): return np.isclose(x[0], 1.0)

fdim = mesh.topology.dim - 1

left_facets = locate_entities_boundary(mesh, fdim, left_boundary)
right_facets = locate_entities_boundary(mesh, fdim, right_boundary)

facet_indices = np.concatenate([left_facets, right_facets])
facet_markers = np.concatenate([np.full(len(left_facets), 1, dtype=np.int32),
                                np.full(len(right_facets), 2, dtype=np.int32)])
facet_tags = meshtags(mesh, fdim, facet_indices, facet_markers)
print("Facet indices: ", facet_indices, flush=True)
print("Facet markers: ", facet_markers, flush=True)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Finite elements for the velocity
Pk_vec = element("Lagrange", mesh.basix_cell(), degree=k, shape=(mesh.geometry.dim,)) # Vector Lagrange elements of order k
V = dfx.fem.functionspace(mesh, Pk_vec) # Velocity function space
u = dfx.fem.Function(V) # velocity
u.x.array[:] = 1 # Set velocity u=1.0 everywhere

# Finite elements for the concentration
Pk = element("Lagrange", mesh.basix_cell(), degree=k) # Scalar Lagrange elements of order k
W = dfx.fem.functionspace(mesh, Pk) # Concentration function space
print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

# Trial and test functions
c, w = ufl.TrialFunction(W), ufl.TestFunction(W)

# Functions for storing solution
c_h  = dfx.fem.Function(W) # Concentration at current  timestep
c_   = dfx.fem.Function(W) # Concentration at previous timestep

#------VARIATIONAL FORM------#
D = dfx.fem.Constant(mesh, dfx.default_scalar_type(D_value)) # Diffusion coefficient
f = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0)) # Source term 
deltaT = dfx.fem.Constant(mesh, dfx.default_scalar_type(dt)) # Form-compiled timestep        
hf = ufl.CellDiameter(mesh) # Cell diameter

# Upwind velocity 
un = (dot(u, n) + abs(dot(u, n))) / 2.0

# Bilinear form
a0 = c * w / deltaT * dx # Time derivative
a1 = dot(grad(w), D * grad(c) - u * c) * dx # Flux term integrated by parts


# Sum all bilinear form terms and compile form
a = a0+a1

# Linear form
L = c_ * w / deltaT * dx + w * f * dx # Time derivative term + source term

# Set strong Dirichlet BCs left and right-hand side
bc_left  = dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0))
# bc_right = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
dof_left  = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0.0))
# dof_right = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 1.0))
bcs = [dfx.fem.dirichletbc(bc_left, dof_left, W),
    #    dfx.fem.dirichletbc(bc_right, dof_right, W)
]

## Robin BCs
u_ex = lambda x: 1 + 0.0000001 *x[0] #x[0]**2 + 2*x[1]**2  
x = ufl.SpatialCoordinate(mesh)
s = Constant(mesh, dfx.default_scalar_type(0.5))
r = Constant(mesh, dfx.default_scalar_type(100)) # for heat transfer, models the heat transfer coefficient
g = dot(n, grad(u_ex(x))) # corresponding to the Neumann BC

a += r * inner(c, w) * ds(RIGHT)
L += r * inner(s, w) * ds(RIGHT)

a_cpp = dfx.fem.form(a)
L_cpp = dfx.fem.form(L)

# Compile total concentration integral for calculation in time-loop
total_c_form = dfx.fem.form(c_h * dx)

# Assemble linear system
A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()

b = create_vector(L_cpp) # Create RHS vector

# Configure direct solver
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType('preonly')
solver.getPC().setType('lu')
solver.getPC().setFactorSolverType('mumps')
solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization


def assemble_transport_RHS():
    """ Assemble the right-hand side of the variational problem. """

    # Zero entries to avoid accumulation
    with b.localForm() as b_loc: b_loc.set(0)

    # Assemble vector and set BCs
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # MPI communication
    set_bc(b, bcs=bcs)


if __name__ == '__main__':

    snapshots = []
    time_values = []
    x_coords = c_h.function_space.tabulate_dof_coordinates()[:, 0]
    for _ in range(num_timesteps):

        t += dt

        assemble_transport_RHS()

        # Compute solution to the advection-diffusion equation and perform parallel communication
        solver.solve(b, c_h.x.petsc_vec)
        c_h.x.scatter_forward()

        # Update previous timestep
        c_.x.array[:] = c_h.x.array.copy()

        # Print stuff
        print(f"Timestep t = {t}")

        print("Maximum concentration: ", mesh.comm.allreduce(c_h.x.array.max(), op=MPI.MAX))
        print("Minimum concentration: ", mesh.comm.allreduce(c_h.x.array.min(), op=MPI.MIN))

        total_c = dfx.fem.assemble_scalar(total_c_form)
        total_c = mesh.comm.allreduce(total_c, op=MPI.SUM)
        print(f"Total concentration: {total_c:.2e}")

        snapshots.append(c_h.x.array.copy())
        time_values.append(t)
        

    fig, ax = plt.subplots()
    line, = ax.plot(x_coords, snapshots[0])
    ax.set_ylim(min(map(np.min, snapshots)), max(map(np.max, snapshots))+0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("Concentration")
    ax.set_title("Concentration evolution")

    def update(frame):
        line.set_ydata(snapshots[frame])
        ax.set_title(f"Time: {time_values[frame]:2.2f}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=10, blit=False, repeat=False)

    # Plot the solution
    try:
        import matplotlib.pyplot as plt
    except:
        RuntimeError("A matplotlib is required to plot the solution.")
    plt.plot(c_h.function_space.tabulate_dof_coordinates()[:, 0], c_h.x.array)
    plt.show()

    # Save as MP4 (requires ffmpeg installed)
    # ani.save("hal_conservation_0.1.mp4", writer="ffmpeg", fps=20)

    # Or save as GIF (requires imagemagick installed)
    # ani.save("sine_wave.gif", writer="imagemagick", fps=20)

    plt.close()