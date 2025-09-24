# minres on poisson first attempt

using Gridap
using Plots

function minres_poisson(u,N)

f(x) = - Δ(u)(x)

domain = (0.0,1.0,0.0,1.0)
partition = (N,N)
model = CartesianDiscreteModel(domain, partition)

orderV=2
orderU=1

V = FESpace(model,ReferenceFE(lagrangian,Float64,orderV),
            conformity=:H1, dirichlet_tags="boundary")
U = FESpace(model,ReferenceFE(lagrangian,Float64,orderU),
            conformity=:H1, dirichlet_tags="boundary")

Vt = TrialFESpace(V,0.0)
Ut = TrialFESpace(U,0.0)

Y = MultiFieldFESpace([Vt,Ut])
X = MultiFieldFESpace([V,U])

tri = Triangulation(model)
degree = 8
dΩ = Measure(tri,degree)

#first pass with full norm
a(u,v) = ∫(u*v + ∇(u)⋅∇(v))dΩ

b((ε,u),(v,z))= a(ε,v) + ∫(∇(v)⋅∇(u) + ∇(ε)⋅∇(z))dΩ

l((v,z))=∫(f*v)dΩ

op = AffineFEOperator(b,l,X,Y)
xh = solve(op)
εh, uh = xh

e = u - uh

  erru = sqrt(sum(a(e,e)))
  erreps = sqrt(sum(a(εh,εh)))

  return erru, erreps

end #function





function h_refinement(u,ncells)

  erru_h = []
  erreps_h = []
  Ns = []

    for N in ncells

      erru, erreps = minres_solver(u,N)
      push!(erru_h,erru)
      push!(erreps_h,erreps)

      push!(Ns,N)

    end

  return erru_h, erreps_h, Ns

end



    
function convergence_plot(Narr,errors1, errors2)
  plot(Narr,errors1,shape=:auto)
  plot!(xaxis=:log, yaxis=:log,
  shape=:auto,
  label=["errors"],
  xlabel="N",ylabel="error norm")
  plot!(Narr,errors2,shape=:auto)
end

# first experiment!
	u_1(x) = (x[1]^2-x[1])*(x[2]^2-x[2])
	ncells_1 = [ 2^i for i in 2:8 ]
	erru_1, erreps_1, Ns_1 = h_refinement(u_1,ncells_1)  
  
  convergence_plot(Ns_1, erru_1, erreps_1)


