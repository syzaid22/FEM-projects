"diffusion only case"
u_exact(x) = sin(pi*x) 
μ(x) = 1.0 + x
β(x) = 0.0 # no convection
τ = 0.0 # streamline diffusion off
f(x) = (pi^2)*(1+x)*sin(pi*x) - pi*cos(pi*x)
domain_min = 0.0
domain_max = 1.0

p = 1 # order of FE 
N_array = [ 2^n for n = 2:5]

errL2_h, errH1_h, h_array = h_refinement_test(p,N_array,u_exact,f,μ,β,domain_min,domain_max; τ=τ)

convergence_plot(h_array,errL2_h,errH1_h)

