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

mesh = UnitSquareMesh(40,40)
n = FacetNormal(mesh)

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

E = Constant(1.e3)
nu = Constant(1./3.)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
              
c0    = Constant(4.0e-10) # 1/Pa
kappa = Constant(5.1e-8) # m^2
alpha = Constant(0.9)     # Biot-Willis
muf   = Constant(1.e-3)  # Pa.s
rho   = Constant(1.)      # kg/m^3

ff    = Constant((0.0,0.0))
mp    = Constant(0)
tract = Constant((0,-1.e2)) # base traction

dt = 0.01; inc = 0; t = 0.  #freq = 1; 

Vh = VectorElement("CG", mesh.ufl_cell(), 1)
Qh = FiniteElement("CG", mesh.ufl_cell(), 1)
Hh = FunctionSpace(mesh,MixedElement([Vh,Qh]))

Sol = Function(Hh);
u, p = TrialFunctions(Hh)
v, q = TestFunctions(Hh)

uold   = Function(Hh.sub(0).collapse())
pold   = Function(Hh.sub(1).collapse())

# ******* Output file  ********** #

fileO = XDMFFile(mesh.mpi_comm(), "out/PrimalMandelTimeDepCastell.xdmf")
fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

bcU1 = DirichletBC(Hh.sub(0).sub(0), Constant(0), bdry, left)
bcU2 = DirichletBC(Hh.sub(0).sub(1), Constant(0), bdry, bot)
bcP = DirichletBC(Hh.sub(1), Constant(0), bdry, right)
bcs = [bcU1,bcU2,bcP]

PLeft = 2*mu*inner(eps(u),eps(v)) * dx \
    + lmbda*div(u)*div(v) * dx \
    - alpha*p*div(v) * dx \
    + 1./dt*(c0*p + alpha*div(u))*q*dx \
    + dot(kappa/muf*grad(p),grad(q)) * dx

PRight = dot(rho*ff,v)*dx \
    + 1./dt*(c0*pold + alpha*div(uold))*q*dx \
    + dot(tract,v)*ds(top)


while (inc <= 100):
    
    print("t=%.3f" % t)
    
    solve(PLeft == PRight, Sol, bcs)
    u_h,p_h = Sol.split()

    u_h.rename("u","u"); fileO.write(u_h, t)
    p_h.rename("p","p"); fileO.write(p_h, t)
        
    assign(uold,u_h); assign(pold,p_h)

    t += dt; inc += 1 
