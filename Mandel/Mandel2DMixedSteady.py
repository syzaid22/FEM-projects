from dolfin import *

'''
 -------------------------------------------------------

 * Mandel's test. 

 * Boundary conditions:
 
   sigma * n = t on top 
   sigma * n = 0 on right
         u.n = 0 on bottom U left  
           p = 0 on right
         z.n = 0 on bottom U left U top 


need to plot var over time at the top left corner: 
u1 and p and strain, etc from t = 0 until t = 100*dt

then need to plot over line, much later: 


 
------------------------------------------------------- 
'''

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

  
CinvTimes = lambda s: 0.5/mu * s - lmbda/(2.*mu*(2*lmbda+2.*mu))*tr(s)*Identity(2)


I = Constant(((1,0),(0,1)))

mesh = UnitSquareMesh(30,30)
n = FacetNormal(mesh)
tan = as_vector((-n[1],n[0]))
bdry = MeshFunction("size_t", mesh, 1)
bdry.set_all(0)
Left  = CompiledSubDomain("near(x[0],0) && on_boundary")
Right = CompiledSubDomain("near(x[0],1) && on_boundary")
Bot   = CompiledSubDomain("near(x[1],0) && on_boundary")
Top   = CompiledSubDomain("near(x[1],1) && on_boundary")

bot = 91; right = 92; top = 94; left = 96; 
Top.mark(bdry,top); Bot.mark(bdry,bot);
Left.mark(bdry,left); Right.mark(bdry,right);
ds = Measure("ds", subdomain_data=bdry)
 
E = Constant(1.e3)  # Pa
nu = Constant(1./3.)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

c0    = Constant(4.e-10) # 1/Pa
kappa = Constant(5.1e-8) # m^2
alpha = Constant(0.9)     # Biot-Willis
muf   = Constant(1.e-3)  # Pa.s
rho   = Constant(1.)      # kg/m^3

ff    = Constant((0.0,0.0))
mp    = Constant(0)
tract = Constant((0,-1.e2)) # base traction

nitsche = Constant(10.0)

#dt = 0.01; inc = 0; t = 0. 
p_right = Constant(0.0)

l=1 
Pkv = VectorElement('DG', mesh.ufl_cell(), l)
Pk  = FiniteElement('DG', mesh.ufl_cell(), l)
BDM  = FiniteElement('BDM', mesh.ufl_cell(), l+1)
Hh = FunctionSpace(mesh,MixedElement([BDM,BDM,Pk,Pkv,Pk,BDM]))
TTh = TensorFunctionSpace(mesh, 'CG',1)
print("Number of dofs: ", Hh.dim())

Sol = Function(Hh)
sig1,sig2,p,u,gam12,z = TrialFunctions(Hh)
tau1,tau2,q,v,eta12,w = TestFunctions(Hh)

gam = as_tensor(((0,gam12),(-gam12,0)))
eta = as_tensor(((0,eta12),(-eta12,0)))

sig = as_tensor((sig1,sig2)) 
tau = as_tensor((tau1,tau2))  

# ******* Output file  ********** #

fileO = XDMFFile(mesh.mpi_comm(), "out/MandelSteady_BDM.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True
 
bcS1 = DirichletBC(Hh.sub(0), Constant((0,0)), Right)
bcS2 = DirichletBC(Hh.sub(1), Constant((0,0)), Right)
bcS1t = DirichletBC(Hh.sub(0), Constant((0,0)), Top)
bcS2t = DirichletBC(Hh.sub(1), tract,           Top)
bcZ1  = DirichletBC(Hh.sub(5), Constant((0,0)), Top)
bcZ2  = DirichletBC(Hh.sub(5), Constant((0,0)), Bot)
bcZ3  = DirichletBC(Hh.sub(5), Constant((0,0)), Left)

bcs = [bcS1,bcS2,bcS1t,bcS2t,bcZ1,bcZ2,bcZ3]


A1  = inner(CinvTimes(sig),tau)*dx 
B1t = dot(u,div(tau)) * dx + inner(gam,tau) * dx
B1  = dot(v,div(sig)) * dx + inner(eta,sig) * dx

A2 = 1.0/kappa*dot(z,w)*dx
B2t = p*div(w)*dx
B2 = q*div(z)*dx 
C2 = (c0+2*alpha**2/(2*mu+2*lmbda))*p*q*dx

Dt = alpha/(2*mu+2*lmbda)*tr(tau)*p*dx
D =  alpha/(2*mu+2*lmbda)*tr(sig)*q*dx

LHS = A1  +  B1t     + Dt \
    + B1                  \
    +             A2 + B2t \
    - D          + B2 - C2 \
    - dot(tau1,n)*tan[0]*dot(u,tan)*ds((left,bot)) \
    - dot(tau2,n)*tan[1]*dot(u,tan)*ds((left,bot))
    #- dot(tau*n,tan)*dot(u,tan) * ds((left,bot)) 
      


# postulate: <tau*n,u> = <tau*n . n , u.n> + <(tau*n).t, u.t>
# as we are imposing u.n = 0 (naturally) then only the other term remains 

RHS =  - dot(rho*ff,v) * dx \
    - mp*q*dx \
    - p_right*dot(w,n)*ds(right) 

solve(LHS == RHS, Sol, bcs, solver_parameters = {"linear_solver": "mumps"})
 
sig1,sig2,p,u,gam12,z = Sol.split()
 
sigh = project(as_tensor((sig1,sig2)), TTh)

u.rename("u","u"); fileO.write(u, 0.0)
p.rename("p","p"); fileO.write(p, 0.0)
sigh.rename("sig","sig"); fileO.write(sigh,0.0)
z.rename("z","z"); fileO.write(z,0.0)
      