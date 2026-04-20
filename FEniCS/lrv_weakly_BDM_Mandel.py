from fenics import *

'''
 -------------------------------------------------------

 * Mandel's test. 

 * Boundary conditions:
 
   sigma * n = t on top 
   sigma * n = 0 on right
         u.n = 0 on bottom U left  
           p = 0 on right
k/muf*grad(pP).n = 0 on bottom U left U top 


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


eps = lambda v: sym(grad(v))
Ctimes = lambda d: 2*mu*d + lmbda*tr(d)*I
kappa = lambda d,p: kap0#*3*exp(30*(c0*p+alpha*tr(d)))  

I = Constant(((1,0),(0,1)))

mesh = UnitSquareMesh(50,50)
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

'''
mu too big. Need to try with smaller in primal linear 
'''

E = Constant(1.e3)  # Pa
nu = Constant(1./3.)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

c0    = Constant(4.e-10) # 1/Pa
kap0  = Constant(5.1e-8) # m^2
alpha = Constant(0.9)     # Biot-Willis
muf   = Constant(1.e-3)  # Pa.s
rho   = Constant(1.)      # kg/m^3

ff    = Constant((0.0,0.0))
mp    = Constant(0)
tract = Constant((0,-1.e2)) # base traction

dt = 0.01; inc = 0; t = 0. 

l=1 
P1 = FiniteElement("CG", mesh.ufl_cell(), l+1)
Pkv = VectorElement('DG', mesh.ufl_cell(), l)

Pk  = FiniteElement('DG', mesh.ufl_cell(), l)
BDM  = FiniteElement('BDM', mesh.ufl_cell(), l+1)
Hh = FunctionSpace(mesh,MixedElement([BDM,BDM,P1,BDM,BDM,Pkv,P1]))
TTh = TensorFunctionSpace(mesh, 'CG',1)

Sol = Function(Hh);
dSol = TrialFunction(Hh)
d1,d2,p,sig1,sig2,u,gam12 = split(Sol)
e1,e2,q,tau1,tau2,v,eta12 = TestFunctions(Hh)


gam = as_tensor(((0,gam12),(-gam12,0)))
eta = as_tensor(((0,eta12),(-eta12,0)))

sig = as_tensor((sig1,sig2)) 
tau = as_tensor((tau1,tau2)) 
d = as_tensor((d1,d2))
e = as_tensor((e1,e2))


d1_old = Function(Hh.sub(0).collapse())
d2_old = Function(Hh.sub(1).collapse())
dold   = as_tensor((d1_old,d2_old))
pold   = Function(Hh.sub(2).collapse())

# ******* Output file  ********** #

fileO = XDMFFile(mesh.mpi_comm(), "out/WeaklyMandelTimeDep.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

'''
   sigma * n = t on top 
   sigma * n = 0 on right
         u.n = 0 on bottom U left  
           p = 0 on right
k/muf*grad(pP).n = 0 on bottom U left U top 
'''

bcP = DirichletBC(Hh.sub(2), Constant(0), Right)
bcS1 = DirichletBC(Hh.sub(3), Constant((0,0)), Right)
bcS2 = DirichletBC(Hh.sub(4), Constant((0,0)), Right)
bcS3 = DirichletBC(Hh.sub(3), Constant((0,0)), Top)
bcS4 = DirichletBC(Hh.sub(4), tract,           Top)
bcs = [bcP,bcS1,bcS2,bcS3,bcS4]

# (s1 s2 / s3 s4) (0 1) = s2 / s4 = (0,-a) 

A   = inner(Ctimes(d),e)*dx \
    + kappa(d,p)/muf*dot(grad(p),grad(q))*dx \
    + c0/dt * p*q*dx \
    + alpha/dt*q*tr(d)*dx \
    - alpha*p*tr(e)*dx
    
B1t = - inner(sig,e) * dx
B1  = - inner(tau,d) * dx
B2t = - dot(u,div(tau)) * dx - inner(gam,tau) * dx
B2  = - dot(v,div(sig)) * dx - inner(eta,sig) * dx

LHS = A  +  B1t      \
    + B1      + B2t\
    +     B2       

# postulate: <tau*n,u> = <tau*n . n , u.n> + <(tau*n).t, u.t>
# as we are imposing u.n = 0 (naturally) then only the other term remains 

RHS = -dot(tau*n,tan)*dot(u,tan) * ds((left,bot)) \
    + dot(rho*ff,v) * dx \
    + c0/dt * pold*q*dx \
    + alpha/dt*q*tr(dold)*dx \
    + mp * q * dx 

FF = LHS - RHS

Tang = derivative(FF,Sol,dSol)
problem = NonlinearVariationalProblem(FF, Sol, bcs, J=Tang)
solver  = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver']                    = 'newton'
solver.parameters['newton_solver']['linear_solver']      = 'mumps'
#solver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
#solver.parameters['newton_solver']['absolute_tolerance'] = 4.67e-6
solver.parameters["newton_solver"]["convergence_criterion"] = 'incremental'
#solver.parameters['newton_solver']['relative_tolerance'] = 1e-5
solver.parameters['newton_solver']['maximum_iterations'] = 10

while (inc <= 100):
    
    print("t=%.3f" % t)
    
    solver.solve()
    d1,d2,p,sig1,sig2,u,gam12 = Sol.split()

    sigh = project(as_tensor((sig1,sig2)), TTh)
    dh = project(as_tensor((d1,d2)), TTh)
    
    #if (inc <= 50 or inc == 60 or inc == 5000): I'll save them all. Only 100
    u.rename("u","u"); fileO.write(u, t)
    p.rename("p","p"); fileO.write(p, t)
    sigh.rename("sig","sig"); fileO.write(sigh,t)
    dh.rename("d","d"); fileO.write(dh,t)
        
    assign(d1_old,d1); assign(d2_old,d2); assign(pold,p)

    dold = as_tensor((d1_old,d2_old))
    
    t += dt; inc += 1 
