'''
Convergence test in 2D 

c0*p - div(z)          = g
 1/kappa*zeta - grad p = f


Unit square, manufactured solutions
'''


from fenics import *
import sympy2fenics as sf
from petsc4py import PETSc

PETScOptions.set("mat_mumps_icntl_4", 1) # 1-2, verbosity
PETScOptions.set("mat_mumps_icntl_14", 1000) # memory estimate
PETScOptions.set("mat_mumps_icntl_22", 1)
PETScOptions.set("mat_mumps_icntl_24", 1)
PETScOptions.set("mat_mumps_icntl_13", 1) # accept almost zero pivots
PETScOptions.set("mat_mumps_cntl_1", 1e-8)



parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4


fileO = XDMFFile("out/Convergence2D_dummy.xdmf")
fileO.parameters['rewrite_function_mesh']=True
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True



str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))


# ********* model constants  ******* #
d      = 2
k      = 1

kappa  = Constant(1.)
c0     = Constant(1.)


# ******* Exact solutions for error analysis ****** #
z_str = '(sin(x),cos(x)*sin(y))'
p_str = 'sin(pi*x)*sin(pi*y)'

nkmax = 6; 

hh = []; nn = []; ez = []; rz = [];  ep = []; rp = []; mass= []

rp.append(0.0); rz.append(0.0);


# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1); mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh); hh.append(mesh.hmax())

    bdry = MeshFunction("size_t", mesh, d-1)
    bdry.set_all(0)
    Gamma = CompiledSubDomain("(near(x[0],0) || near(x[1],0)) && on_boundary")
    Sigma = CompiledSubDomain("(near(x[0],1) || near(x[1],1)) && on_boundary")

    GM = 91; SG = 92;
    Gamma.mark(bdry,GM); Sigma.mark(bdry,SG)
    ds = Measure("ds", subdomain_data=bdry)
    
    p_ex = Expression(str2exp(p_str),degree=6+k, domain=mesh)
    z_ex = Expression(str2exp(z_str),degree=6+k, domain=mesh)

    f_ex = 1/kappa*z_ex - grad(p_ex)
    
    g_ex   = c0*p_ex - div(z_ex)
    
    # ********* Finite dimensional spaces ********* #
    
    RT  = FiniteElement('RT', mesh.ufl_cell(), k+1)
    Pk  = FiniteElement('DG', mesh.ufl_cell(), k)

    Hh = FunctionSpace(mesh, MixedElement([RT,Pk]))

    print("**************** Total Dofs = ", Hh.dim())

    nn.append(Hh.dim())
    
    Sol = Function(Hh); 
    z, p = TrialFunctions(Hh)
    w, q = TestFunctions(Hh)


    # Essential BCs for sig on Sigma and for phi on Gamma
    
    zGam = project(z_ex,Hh.sub(0).collapse())
    
    bcZ = DirichletBC(Hh.sub(0),zGam,bdry,GM)
    
    # Natural BCs for p on Sigma 
    
    # ********  Weak forms ********** #
    
    a   = 1/kappa*dot(z,w)*dx
    bt  = p*div(w)*dx
    b   = q*div(z)*dx
    c   = c0*p*q*dx



    LHS = a + bt + b - c

    RHS = dot(f_ex,w) * dx \
        - g_ex*q*dx + p_ex*dot(w,n)*ds(SG) 
    
    solve(LHS == RHS, Sol, bcs = bcZ, \
          solver_parameters={"linear_solver":'mumps'})

    z, p  = Sol.split()

    ez.append(pow(assemble((z_ex-z)**2*dx + div(z_ex-z)**2*dx),0.5))
    ep.append(errornorm(p_ex,p,'L2'))
    
    mass_h = project(g_ex - c0*p + div(z),Hh.sub(1).collapse())

    mass.append(norm(mass_h.vector(),'linf'))

    z.rename("z","z"); fileO.write(z, 1.0*nk)
    p.rename("p","p"); fileO.write(p, 1.0*nk)
    if(nk>0):

        rz.append(ln(ez[nk]/ez[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
        

# ********  Generating error history **** #
print('=================================================================')
print('   nn     hh     e(z)     r(z)    e(p)    r(p)   mass_h')
print('=================================================================')

for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:1.1e} & {:.2f} & {:1.1e} & {:.2f} & {:1.2e} '.format(nn[nk], hh[nk], ez[nk], rz[nk], ep[nk], rp[nk], mass[nk]))

print('=================================================================')
 

