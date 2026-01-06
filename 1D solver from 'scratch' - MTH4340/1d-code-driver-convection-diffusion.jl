"convection-diffusion"
u_c(x) = sin(pi * x)
μ_c(x) = 1.0 + x
β_c(x) = 2.0 * x
τ_c = 1.0 # τ=0.0 switches off streamline diffusion; τ≠0.0 switches it on
f_c(x) = (pi^2*(1 + x)*sin(pi*x)) + (pi * cos(pi * x) * (2.0 * x - 1.0))
domain_min_c = 0.0
domain_max_c = 1.0

p_c = 1 # order of FE
N_array_c = [ 2^n for n = 2:5]

errL2_h_c, errH1_h_c, h_array_c = h_refinement_test(p_c,N_array_c,u_c,f_c,μ_c,β_c,domain_min_c,domain_max_c; τ=τ_c)

convergence_plot(h_array_c,errL2_h_c,errH1_h_c)
