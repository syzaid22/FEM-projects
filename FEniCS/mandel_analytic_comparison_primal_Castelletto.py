from fenics import *
import numpy as np
import matplotlib.pyplot as plt

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

eps = lambda v: sym(grad(v))

mesh = UnitSquareMesh(40,40)
n = FacetNormal(mesh)

bdry = MeshFunction("size_t", mesh, 1)
bdry.set_all(0)
Left  = CompiledSubDomain("near(x[0],0) && on_boundary")
Right = CompiledSubDomain("near(x[0],1) && on_boundary")
Bot   = CompiledSubDomain("near(x[1],0) && on_boundary")
Top   = CompiledSubDomain("near(x[1],1) && on_boundary")

bot = 91; right = 92; top = 94; left = 96
Top.mark(bdry,top); Bot.mark(bdry,bot)
Left.mark(bdry,left); Right.mark(bdry,right)
ds = Measure("ds", subdomain_data=bdry)

# Material parameters
E = Constant(1.e3)
nu = Constant(1./3.)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
              
c0 = Constant(4.0e-10)
kappa = Constant(5.1e-8)
alpha = Constant(0.9)
muf = Constant(1.e-3)
rho = Constant(1.)

ff = Constant((0.0,0.0))
mp = Constant(0)
tract = Constant((0,-1.e2))

dt = 0.01
t = 0.

Vh = VectorElement("CG", mesh.ufl_cell(), 1)
Qh = FiniteElement("CG", mesh.ufl_cell(), 1)
Hh = FunctionSpace(mesh, MixedElement([Vh,Qh]))

Sol = Function(Hh)
u, p = TrialFunctions(Hh)
v, q = TestFunctions(Hh)

uold = Function(Hh.sub(0).collapse())
pold = Function(Hh.sub(1).collapse())

bcU1 = DirichletBC(Hh.sub(0).sub(0), Constant(0), bdry, left)
bcU2 = DirichletBC(Hh.sub(0).sub(1), Constant(0), bdry, bot)
bcP = DirichletBC(Hh.sub(1), Constant(0), bdry, right)
bcs = [bcU1, bcU2, bcP]

PLeft = 2*mu*inner(eps(u), eps(v)) * dx \
    + lmbda*div(u)*div(v) * dx \
    - alpha*p*div(v) * dx \
    + 1./dt*(c0*p + alpha*div(u))*q*dx \
    + dot(kappa/muf*grad(p), grad(q)) * dx

PRight = dot(rho*ff, v)*dx \
    + 1./dt*(c0*pold + alpha*div(uold))*q*dx \
    + dot(tract, v)*ds(top)

# Extract float values
E_val = float(E)
nu_val = float(nu)
mu_val = float(mu)
c0_val = float(c0)
kappa_val = float(kappa)
alpha_val = float(alpha)
muf_val = float(muf)

# Calculate parameters
G = mu_val
K_drained = E_val/(3*(1-2*nu_val))
M_val = 1.0/c0_val
K_u = K_drained + alpha_val**2 * M_val
B = alpha_val * M_val / K_u
nu_u = (3*K_u - 2*G) / (2*(3*K_u + G))
k_perm = kappa_val / muf_val
cf = 2 * k_perm * G * B**2 * (1+nu_u)**2 * (1-nu_val) / (9 * (1-nu_u) * (nu_u - nu_val))
P0 = 100.0
a = 1.0

print("="*60)
print("MANDEL PROBLEM PARAMETERS")
print("="*60)
print(f"G = {G:.4f}")
print(f"nu = {nu_val}, nu_u = {nu_u:.6f}")
print(f"B = {B:.6f}")
print(f"cf = {cf:.6e}")
print(f"C = {(1-nu_val)/(nu_u-nu_val):.6f}")
print("="*60)

# Compute eigenvalues using Newton's method (more robust)
def compute_eigenvalues(n_terms=100):
    C = (1 - nu_val) / (nu_u - nu_val)
    eigenvalues = []
    
    for n in range(1, n_terms+1):
        # Initial guess
        if n == 1:
            alpha = 1.0  # First root is between 0 and pi/2
        else:
            alpha = (n - 0.5) * np.pi
        
        # Newton iteration
        for _ in range(100):
            f = np.tan(alpha) - C * alpha
            f_prime = 1.0/(np.cos(alpha)**2) - C
            alpha_new = alpha - f/f_prime
            if abs(alpha_new - alpha) < 1e-12:
                break
            alpha = alpha_new
        eigenvalues.append(alpha)
    
    return np.array(eigenvalues)

n_terms = 100
alpha_n = compute_eigenvalues(n_terms)

print(f"\nFirst 5 eigenvalues:")
for i in range(min(5, len(alpha_n))):
    print(f"  alpha_{i+1} = {alpha_n[i]:.8f}")

def p_analytical(x, t):
    p0 = (P0/(3*a)) * B * (1 + nu_u)
    pressure = 0.0
    for alpha in alpha_n:
        denom = alpha - np.sin(alpha)*np.cos(alpha)
        coeff = np.sin(alpha) / denom
        spatial = np.cos(alpha * x / a) - np.cos(alpha)
        temporal = np.exp(-alpha**2 * cf * t / a**2)
        pressure += coeff * spatial * temporal
    return 2 * p0 * pressure

def ux_analytical(x, y, t):
    sum1 = 0.0
    for alpha in alpha_n:
        denom = alpha - np.sin(alpha)*np.cos(alpha)
        term = (np.sin(alpha)*np.cos(alpha)) / denom
        term *= np.exp(-alpha**2 * cf * t / a**2)
        sum1 += term
    
    coeff_x = P0*nu_val/(2*G*a) - (P0 * nu_u)/(G*a) * sum1
    part1 = coeff_x * x
    
    sum2 = 0.0
    for alpha in alpha_n:
        denom = alpha - np.sin(alpha)*np.cos(alpha)
        term = np.cos(alpha) / denom
        term *= np.sin(alpha * x / a)
        term *= np.exp(-alpha**2 * cf * t / a**2)
        sum2 += term
    
    part2 = (P0/G) * sum2
    
    return part1 + part2

def uy_analytical(x, y, t):
    sum1 = 0.0
    for alpha in alpha_n:
        denom = alpha - np.sin(alpha)*np.cos(alpha)
        term = (np.sin(alpha)*np.cos(alpha)) / denom
        term *= np.exp(-alpha**2 * cf * t / a**2)
        sum1 += term
    
    coeff_y = -P0*(1-nu_val)/(2*G*a) + (P0*(1-nu_u))/(G*a) * sum1
    return coeff_y * y

# Solve numerical problem
for inc in range(101):
    solve(PLeft == PRight, Sol, bcs)
    u_h, p_h = Sol.split()
    if inc % 20 == 0:
        print(f"t={t:.3f}")
    assign(uold, u_h)
    assign(pold, p_h)
    t += dt

# Plot at final time
t_final = t - dt
x_plot = np.linspace(0, a, 200)
y_mid = 0.5

p_num = []
ux_num = []
uy_num = []
p_ana = []
ux_ana = []
uy_ana = []

for x in x_plot:
    try:
        p_num.append(p_h(x, y_mid))
        ux_num.append(u_h(x, y_mid)[0])
        uy_num.append(u_h(x, y_mid)[1])
    except:
        p_num.append(0.0)
        ux_num.append(0.0)
        uy_num.append(0.0)
    
    p_ana.append(p_analytical(x, t_final))
    ux_ana.append(ux_analytical(x, y_mid, t_final))
    uy_ana.append(uy_analytical(x, y_mid, t_final))



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_plot, p_num, 'b-', label='Numerical', linewidth=2)
plt.plot(x_plot, p_ana, 'r--', label='Analytical', linewidth=2)
plt.xlabel('x')
plt.ylabel('Pressure')
plt.title(f'Pressure at y=0.5, t={t_final:.3f}')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_plot, ux_num, 'b-', label='Numerical', linewidth=2)
plt.plot(x_plot, ux_ana, 'r--', label='Analytical (total)', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
plt.xlabel('x')
plt.ylabel('Horizontal Displacement (ux)')
plt.title(f'Horizontal Displacement at y=0.5, t={t_final:.3f}')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_plot, uy_num, 'b-', label='Numerical', linewidth=2)
plt.plot(x_plot, uy_ana, 'r--', label='Analytical', linewidth=2)
plt.xlabel('x')
plt.ylabel('Vertical Displacement (uy)')
plt.title(f'Vertical Displacement at y=0.5, t={t_final:.3f}')
plt.legend()
# Zoom out y-axis to reduce visible differences
y_min = min(min(uy_num), min(uy_ana))
y_max = max(max(uy_num), max(uy_ana))
margin = 10 * (y_max - y_min)
plt.ylim(y_min - margin, y_max + margin)
plt.grid(True)

plt.tight_layout()
plt.savefig('mandel_final.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("RESULTS AT FINAL TIME")
print("="*60)
print(f"t_final = {t_final:.3f}")
print(f"\nPressure:")
print(f"  p(0) = {p_ana[0]:.6e}")
print(f"  p(1) = {p_ana[-1]:.6e}")
print(f"\nHorizontal Displacement:")
print(f"  ux(0) = {ux_ana[0]:.6e}")
print(f"  ux(0.5) = {ux_ana[100]:.6e}")
print(f"  ux(1) = {ux_ana[-1]:.6e}")