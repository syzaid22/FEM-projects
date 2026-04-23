"""
Convergence test using a single fine‑mesh reference solution.
Errors are computed on each coarse mesh by projecting the fine solution onto the coarse space.
Mixed formulation: c0*p - div(z) = g,  1/kappa*z - grad(p) = f
Constant f = (1,1), g = 1.
BCs: Dirichlet z = (1,1) on Gamma (x=0 or y=0)
      Neumann p = 1 on Sigma (x=1 or y=1)
"""

from fenics import *
from petsc4py import PETSc
import numpy as np

PETScOptions.set("mat_mumps_icntl_4", 1)
PETScOptions.set("mat_mumps_icntl_14", 1000)
PETScOptions.set("mat_mumps_icntl_22", 1)
PETScOptions.set("mat_mumps_icntl_24", 1)
PETScOptions.set("mat_mumps_icntl_13", 1)
PETScOptions.set("mat_mumps_cntl_1", 1e-8)

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4

# ------------------------------------------------------------
# Model constants
kappa = Constant(1.)
c0    = Constant(1.)
k     = 0                     # polynomial degree: RT_{k+1} x DG_k

p_Neumann_val   = Constant(0.)         # Neumann BC for p on Sigma

nkmax = 4
fine_level = nkmax+3

def solve_on_mesh(mesh):
    """Solve the mixed problem on a given mesh.
       Returns (z, p, V_RT, V_DG) where V_RT, V_DG are the function spaces.
    """
    d = mesh.geometry().dim()
    n = FacetNormal(mesh)
    bdry = MeshFunction("size_t", mesh, d-1)
    bdry.set_all(0)
    Gin = CompiledSubDomain("(near(x[0],0)) && on_boundary")
    Gout = CompiledSubDomain("(near(x[0],1)) && on_boundary")
    Gwall = CompiledSubDomain("(near(x[1],1) || near(x[1],0)) && on_boundary")
    
    GI,GO,GW  = 91, 92, 93
    Gin.mark(bdry, GI)
    Gout.mark(bdry, GO)
    Gwall.mark(bdry, GW)
    
    ds = Measure("ds", subdomain_data=bdry)

    # Mixed function space
    RT = FiniteElement('BDM', mesh.ufl_cell(), k+1)
    Pk = FiniteElement('DG', mesh.ufl_cell(), k)
    Hh = FunctionSpace(mesh, MixedElement([RT, Pk]))
    V_RT = Hh.sub(0).collapse()
    V_DG = Hh.sub(1).collapse()

    z, p = TrialFunctions(Hh)
    w, q = TestFunctions(Hh)

    f = Constant((1,0)) #Expression(('sin(pi*x[0])', 'cos(pi*x[1])'), degree=k+2, domain = mesh)
    g = Constant(1) #Expression('sin(pi*x[0])*sin(pi*x[1])', degree=k+2, domain = mesh)

    z_in = Constant((1,0)) #Expression(('5*x[1]*(1-x[1])', '0'), degree=k+2, domain = mesh)
    # Dirichlet BC for z on Gamma
    bcZ1 = DirichletBC(Hh.sub(0), z_in, bdry, GI)
    bcZ2 = DirichletBC(Hh.sub(0), Constant((0,0)), bdry, GW)
    bcs = [bcZ1,bcZ2]
    
    # Weak form
    a = 1/kappa*dot(z, w)*dx
    bt = p*div(w)*dx
    b = q*div(z)*dx
    c = c0*p*q*dx
    LHS = a + bt + b - c
    RHS = dot(f, w)*dx - g*q*dx + p_Neumann_val*dot(w, n)*ds(GO)

    Sol = Function(Hh)
    solve(LHS == RHS, Sol, bcs=bcs,
          solver_parameters={"linear_solver": 'mumps'})
    z_sol, p_sol = Sol.split()
    return z_sol, p_sol, V_RT, V_DG

# ------------------------------------------------------------
# 1. Build and solve on the very fine reference mesh
#print("===== Building fine reference mesh (level {}) =====".format(fine_level))
nps_fine = 2**(fine_level+1)
mesh_fine = UnitSquareMesh(nps_fine, nps_fine)
z_fine, p_fine, V_RT_fine, V_DG_fine = solve_on_mesh(mesh_fine)
print("Fine mesh DOFs (mixed):", V_RT_fine.dim() + V_DG_fine.dim())

# ------------------------------------------------------------
# 2. Loop over coarse levels and compute errors on coarse meshes
hh = []        # coarse mesh sizes
nn = []        # coarse DOFs
ez = []        # H(div) errors (coarse mesh)
rz = []        # H(div) rates
ep = []        # L2 errors for p (coarse mesh)
rp = []        # L2 rates

for i in range(nkmax+1):
    print("\n...... Coarse level i =", i, "......")
    nps_coarse = 2**(i+1)
    mesh_coarse = UnitSquareMesh(nps_coarse, nps_coarse)
    h_coarse = mesh_coarse.hmax()
    hh.append(h_coarse)

    # Solve on coarse mesh
    z_coarse, p_coarse, V_RT_coarse, V_DG_coarse = solve_on_mesh(mesh_coarse)
    nn.append(V_RT_coarse.dim() + V_DG_coarse.dim())

    # Project fine reference solution onto the coarse function spaces
    # This gives functions that live on the coarse mesh and can be compared directly.
    z_fine_on_coarse = interpolate(z_fine, V_RT_coarse)
    p_fine_on_coarse = interpolate(p_fine, V_DG_coarse)

    # Compute errors on the coarse mesh
    # H(div) error for z:
    dz = z_fine_on_coarse - z_coarse
    div_dz = div(z_fine_on_coarse) - div(z_coarse)
    err_z = sqrt( assemble(dot(dz, dz)*dx + div_dz*div_dz*dx) )
    ez.append(err_z)

    # L2 error for p
    dp = p_fine_on_coarse - p_coarse
    err_p = sqrt( assemble(dp*dp*dx))
    
    ep.append(err_p)

    # Compute convergence rates (from second level onward)
    if i > 0:
        r_z = np.log(ez[i]/ez[i-1]) / np.log(hh[i]/hh[i-1])
        r_p = np.log(ep[i]/ep[i-1]) / np.log(hh[i]/hh[i-1])
        rz.append(r_z)
        rp.append(r_p)
    else:
        rz.append(0.0)
        rp.append(0.0)

# ------------------------------------------------------------
# 3. Print error table
print("\n" + "="*50)
print("Convergence against fine‑mesh reference (level {})".format(fine_level))
print("="*50)
print("  i     DOF     h       e(z)       rate(z)      e(p)       rate(p)")
print("="*50)
for i in range(len(nn)):
    print("{:3d}   {:8d}  {:8.4e}  {:8.2e}  {:5.2f}  {:8.2e}   {:5.2f}"
          .format(i, nn[i], hh[i], ez[i], rz[i], ep[i], rp[i]))
print("="*50)
