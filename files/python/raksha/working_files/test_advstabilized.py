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

print = PETSc.Sys.Print

# === Mesh setup ===
if len(argv) < 2:
    raise RuntimeError("Specify number of mesh cells as command line argument.")
N = int(argv[1])
mesh = create_interval(MPI.COMM_WORLD, N, [0.0, 1.0])

# === Facet tagging ===
fdim = mesh.topology.dim - 1

def left_boundary(x): return np.isclose(x[0], 0.0)
def right_boundary(x): return np.isclose(x[0], 1.0)

left_facets = locate_entities_boundary(mesh, fdim, left_boundary)
right_facets = locate_entities_boundary(mesh, fdim, right_boundary)

facet_indices = np.concatenate([left_facets, right_facets])
facet_markers = np.concatenate([np.full(len(left_facets), 1, dtype=np.int32),
                                np.full(len(right_facets), 2, dtype=np.int32)])
facet_tag = meshtags(mesh, fdim, facet_indices, facet_markers)
print("Facet indices: ", facet_indices, flush=True)
print("Facet markers: ", facet_markers, flush=True)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
dS = ufl.Measure("dS", domain=mesh)

# === Simulation parameters ===
D_value = 1e-3
k = 1
t = 0
T = 10
dt = 0.1
num_timesteps = int(T / dt)
n = ufl.FacetNormal(mesh)

# === Function spaces ===
Pk_vec = element("Lagrange", mesh.basix_cell(), degree=k, shape=(mesh.geometry.dim,))
V = functionspace(mesh, Pk_vec)
u = Function(V)
u.x.array[:] = 0.1  # constant velocity field

Pk = element("Lagrange", mesh.basix_cell(), degree=k)
W = functionspace(mesh, Pk)
print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

# === Trial, test, and solution functions ===
c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
c_h = Function(W)
c_ = Function(W)

# === Constants ===
D = Constant(mesh, dfx.default_scalar_type(D_value))
f = Constant(mesh, dfx.default_scalar_type(0.0))
h = Constant(mesh, dfx.default_scalar_type(1.0))
deltaT = Constant(mesh, dfx.default_scalar_type(dt))
beta = Constant(mesh, dfx.default_scalar_type(10.0))
hf = ufl.CellDiameter(mesh)

# === Boundary conditions ===
bc_left = Constant(mesh, dfx.default_scalar_type(1.0))
dof_left = locate_dofs_geometrical(W, lambda x: np.isclose(x[0], 0.0))
bcs = [dirichletbc(bc_left, dof_left, W)]  # Only apply at inlet

# === Variational Form ===
un = (dot(u, n) + abs(dot(u, n))) / 2.0

# Core terms
# a0 = c * w / deltaT * ufl.dx
# a1 = dot(grad(w), D * grad(c) - u * c) * ufl.dx
# a3 = dot(jump(w), un('+') * c('+') - un('-') * c('-')) * dS  # Upwinding

a_time     = c * w / deltaT * ufl.dx
a_advect   = dot(u, grad(c)) * w * ufl.dx
a_diffuse  = dot(grad(c), grad(w)) * D * ufl.dx

a = a_time + a_advect + a_diffuse
L = (c_ / deltaT + f) * w * ufl.dx

# Total forms
# a = a0 + a1 + a3 
# L = c_ * w / deltaT * ufl.dx + w * f * ufl.dx

a_cpp = form(a)
L_cpp = form(L)

# === Total concentration integral ===
total_c_form = form(c_h * ufl.dx)

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

# === RHS assembler ===
def assemble_transport_RHS():
    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs=bcs)

# === Time loop ===
if __name__ == '__main__':

    # Allocate list to store time series snapshots
    snapshots = []
    time_values = []

    # Get x positions once (only for Lagrange P1 elements in 1D)
    x_coords = c_h.function_space.tabulate_dof_coordinates()[:, 0]
    for _ in range(num_timesteps):
        t += dt

        assemble_transport_RHS()
        solver.solve(b, c_h.x.petsc_vec)
        c_h.x.scatter_forward()

        c_.x.array[:] = c_h.x.array.copy()

        print(f"Timestep t = {t:.2f}")
        print("Max conc: ", mesh.comm.allreduce(c_h.x.array.max(), op=MPI.MAX))
        print("Min conc: ", mesh.comm.allreduce(c_h.x.array.min(), op=MPI.MIN))

        total_c = assemble_scalar(total_c_form)
        total_c = mesh.comm.allreduce(total_c, op=MPI.SUM)
        print(f"Total concentration: {total_c:.2e}")

        snapshots.append(c_h.x.array.copy())
        time_values.append(t)

        # from dolfinx.io import XDMFFile

        # with XDMFFile(mesh.comm, f"concentration_{t*100:04f}.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(mesh)
        #     xdmf.write_function(c_h, t)


    # === Plot ===
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required to plot the solution.")

    # plt.plot(c_h.function_space.tabulate_dof_coordinates()[:, 0], c_h.x.array)
    # plt.xlabel("x")
    # plt.ylabel("Concentration")
    # plt.ylim(0, 1.1)
    # plt.title("Concentration profile")
    # plt.grid(True)
    # plt.show()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

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


    plt.show()
