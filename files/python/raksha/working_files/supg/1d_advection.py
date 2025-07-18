from pyclbr import Function
import ufl
import numpy   as np
import dolfinx as dfx

from ufl               import avg, jump, dot, grad, div
from sys               import argv
from mpi4py            import MPI
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
T = 20 # Final time
dt = 0.1 # Timestep size
num_timesteps = int(T / dt)
n = ufl.FacetNormal(mesh)
dx = ufl.Measure("dx", domain=mesh) # Cell integrals

# Function spaces
Pk_vec = element("Lagrange", mesh.basix_cell(), degree=k, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, Pk_vec) # function space for velocity
u = dfx.fem.Function(V) # velocity
Pk = element("Lagrange", mesh.basix_cell(), degree=k)
u.x.array[:] = 0.1 # Set velocity u=1.0 everywhere
W = dfx.fem.functionspace(mesh, Pk) # function space for concentration
deltaT = dfx.fem.Constant(mesh, dfx.default_scalar_type(dt)) # Time step size

print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

# === Trial, test, and solution functions ===
c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
c_h = dfx.fem.Function(W) # concentration at current time step
c_ = dfx.fem.Function(W) # concentration at previous time step

#------VARIATIONAL FORM------#
D = dfx.fem.Constant(mesh, D_value) # Diffusion coefficient
f = dfx.fem.Constant(mesh, 1.0) # Source term
hf = ufl.CellDiameter(mesh) # Cell diameter

u_mag = ufl.sqrt(ufl.dot(u, u) + 1e-10) # Magnitude of velocity
Pe = u_mag * hf / (2 * D) # Peclet number
beta = (ufl.cosh(Pe)/ufl.sinh(Pe))- (1/Pe)
tau = hf * beta/ (2 * u_mag + 1e-10) # Stabilization parameter

# Variational form
a_time     = c * w / deltaT * dx
a_advect   = dot(u, grad(c)) * w * dx
a_diffuse  = dot(grad(c), grad(w)) * D * dx

a = a_time + a_advect + a_diffuse
L = (c_ / deltaT + f) * w * dx

# SUPG terms
residual = dot(u, grad(c_h)) - D * div(grad(c_h)) + (c_h - c_) / deltaT - f
v_supg = tau * dot(u, grad(w))

a_supg = v_supg * (c / deltaT + dot(u, grad(c)) - D * div(grad(c))) * dx
L_supg = v_supg * (c_ / deltaT + f) * dx

a += a_supg
L += L_supg

a_cpp = dfx.fem.form(a)
L_cpp = dfx.fem.form(L)

# === Boundary conditions ===
bc_left = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
dof_left = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0.0))
bcs = [dfx.fem.dirichletbc(bc_left, dof_left, W)]  # Only apply at inlet

bc_right = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))
dof_right = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 1.0))
bcs.append(dfx.fem.dirichletbc(bc_right, dof_right, W))  # Apply at outlet

# === Total concentration integral ===
total_c_form = dfx.fem.form(c_h * dx)

# === Linear system ===
A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()
b = create_vector(L_cpp)

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
solver.setType('preonly')
solver.getPC().setType('lu')
solver.getPC().setFactorSolverType('mumps')
solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1)

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

        # === Compute residual norm ===
        # Step 1: Copy RHS vector
        residual_vec = b.copy()

        # Step 2: Compute A * c_h
        A_ch = b.copy()  # Temporary PETSc vector
        A.mult(c_h.x.petsc_vec, A_ch)

        # Step 3: Subtract A * c_h from b
        residual_vec.axpy(-1.0, A_ch)  # residual_vec = b - A_ch

        # Step 4: Compute residual norm (e.g. L2 norm)
        res_norm = residual_vec.norm()
        print(f"Residual norm ||r|| = {res_norm:.2e}")


        total_c = dfx.fem.assemble_scalar(total_c_form)
        total_c = mesh.comm.allreduce(total_c, op=MPI.SUM)
        print(f"Total concentration: {total_c:.2e}")

        # error_squared = dfx.fem.assemble_scalar(dfx.fem.form(error_form)) # assemble into a scalar, by converting symbolic UFL form to Fenicsx
        # total_residual = mesh.comm.allreduce(error_squared, op=MPI.SUM) # gather all the errors from all processes in case of parallel execution
        # print(f"Total residual: {np.sqrt(total_residual):.2e}")

        snapshots.append(c_h.x.array.copy())
        time_values.append(t)
        

    fig, ax = plt.subplots()
    line, = ax.plot(x_coords, snapshots[0])
    ax.set_ylim(min(map(np.min, snapshots)), max(map(np.max, snapshots))*1.1)
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
    plt.plot(c_h.function_space.tabulate_dof_coordinates()[:, 0], c_h.x.array, label="Numerical solution")
    gamma = (0.1)/D_value
    c_true = (1/0.1) *(x_coords - ((1- np.exp(gamma * x_coords)) / (1 - np.exp(gamma))))
    plt.plot(x_coords, c_true, label="Analytical solution")
    plt.legend()
    plt.show()

    # Save as MP4 (requires ffmpeg installed)
    # ani.save("hal_conservation_0.1.mp4", writer="ffmpeg", fps=20)

    # Or save as GIF (requires imagemagick installed)
    # ani.save("sine_wave.gif", writer="imagemagick", fps=20)

    plt.close()