from __future__ import print_function
from fenics import *
from fenics_adjoint import *
from jet_bcs import JetBCValue, JetBCValueDerivative
import numpy as np


set_log_level(WARNING)

dt = Constant(5E-4)
mu = Constant(1E-3)
rho = Constant(1)

T_term = 5*dt(0)

mesh_file, surface_file = './mesh/mesh.xml', './mesh/mesh_facet_region.xml'
# Load mesh with markers
mesh = Mesh(mesh_file)
surfaces = MeshFunction('size_t', mesh, surface_file)
# NOTE: I want this to be using only 2 surfaces
assert set(surfaces.array()) == set((0, 1, 2, 3, 4, 5, 6))
# Now we can rename 6 to 5
surfaces.array()[np.where(surfaces.array() == 6)[0]] = 5

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define inflow profile
bot = mesh.coordinates().min(axis=0)[1]
top = mesh.coordinates().max(axis=0)[1]

H = top - bot
Um = 1.5

inflow_profile = Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                             '0'), bot=bot, top=top, H=H, Um=Um, degree=2)

# Boundaries
inlet_tag = 3
outlet_tag = 2
wall_tag = 1
cylinder_noslip_tag = 4
jet_tag = 5

# Define boundary conditions
bcu_inlet = DirichletBC(V, inflow_profile, surfaces, inlet_tag)
# No slip
bcu_wall = DirichletBC(V, Constant((0, 0)), surfaces, wall_tag)
bcu_cyl_wall = DirichletBC(V, Constant((0, 0)), surfaces, cylinder_noslip_tag)
# Fixing outflow pressure
bcp_outflow = DirichletBC(Q, Constant(0), surfaces, outlet_tag)

# Now the expression for the jets
radius = 0.05
width = 10
position = 90

# One control Q, the other is -Q
ctrls = [Constant(0.01)]

jet = JetBCValue(radius=radius, width=width, theta0=position, Q=ctrls[-1], degree=1)
jet.user_defined_derivatives = {
    ctrls[-1]: JetBCValueDerivative(radius=radius, width=width, theta0=position, degree=1)
}

bcu_jet = [DirichletBC(V, jet, surfaces, jet_tag)]

# All bcs objects togets
bcu = [bcu_inlet, bcu_wall, bcu_cyl_wall] + bcu_jet
bcp = [bcp_outflow]

# Define trial and test functions
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

# Define functions for solutions at previous and current time steps
# FIXME: this should start from a developed flow, (running init_state until 5)
u_n, u_ = Function(V), Function(V)
p_n, p_ = Function(Q), Function(Q)

comm = mesh.mpi_comm()
XDMFFile(comm, './mesh/u_init.xdmf').read_checkpoint(u_n, 'u0', 0)
XDMFFile(comm, './mesh/p_init.xdmf').read_checkpoint(p_n, 'p0', 0)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))

# Define symmetric gradient
def epsilon(u): return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p): return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / dt, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Time-stepping
t = 0
J = 0
sigma = 2 * Constant(mu) * sym(grad(u_)) - p_ * Identity(2)
# The drag form
ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
n = FacetNormal(mesh)
# NOTE: whatever this ends up being the drag should be over the cylinder
# FIXME: I gues drag**2 + something
drag = dot(dot(sigma, n), Constant((1, 0)))
form = sum(drag**2*ds(tag)
           for tag in (jet_tag, cylinder_noslip_tag))

while t < T_term:
    # Update current time
    t += dt(0)

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    J += assemble(form)

    ctrls.append(Constant(0.01))
    jet.Q = ctrls[-1]
    jet.user_defined_derivatives[ctrls[-1]] = JetBCValueDerivative(
        radius=radius, width=width, theta0=position, degree=1
    )

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

# --------------------------------------------------------------------

ctrls = ctrls[:-1]
Jhat = ReducedFunctional(J, [Control(m) for m in ctrls])

h = [Constant(0.01) for _ in ctrls]
taylor_test(Jhat, ctrls, h)

#exit()
# FIXME: add bounds
#m_opt = minimize(Jhat, options={"maxiter": 2, "disp": True})
#for m in m_opt:
#    print(float(m))
