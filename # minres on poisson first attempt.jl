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



# second pass with "semi"norm
c(u,v) = ∫(∇(u)⋅∇(v))dΩ

d((ε,u),(v,z))= c(ε,v) + ∫(∇(v)⋅∇(u) + ∇(ε)⋅∇(z))dΩ

op2 = AffineFEOperator(d,l,X,Y)
xh2 = solve(op2)
εh2, uh2 = xh2

e2 = u - uh2

 erru2 = sqrt(sum(c(e2,e2)))
 erreps2 = sqrt(sum(c(εh2,εh2)))



nn = Gridap.FESpaces.num_free_dofs(X)

return erru, erreps, erru2, erreps2, nn

end #function





function h_refinement(u,ncells)

  erru_h = []
  erreps_h = []
  erru2_h = []
  erreps2_h = []
  Ns = Float64[]

    for N in ncells

      erru, erreps, erru2, erreps2, nn = minres_poisson(u,N)
      push!(erru_h,erru)
      push!(erreps_h,erreps)

      push!(erru2_h,erru2)
      push!(erreps2_h,erreps2)

      push!(Ns,nn)

    end

  return erru_h, erreps_h, erru2_h, erreps2_h, Ns

end



    
function convergence_plot(Narr,errors1, errors2)
  plot(Narr,errors1,label=["u-uh"],shape=:auto)
  plot!(xaxis=:log, yaxis=:log,
  shape=:auto,
  label=["errors"],
  xlabel="N",ylabel="error norm")
  plot!(Narr,errors2,label=["ϵh"],shape=:auto)
end

  plot!(Narr.^(0.5),Narr.^(-0.5))

# first experiment!
	u_1(x) = (x[1]^2-x[1])*(x[2]^2-x[2])
	ncells_1 = [ 2^i for i in 2:8 ]
	erru_1, erreps_1, erru2_1, erreps2_1, Ns_1 = h_refinement(u_1,ncells_1)  
  
  convergence_plot(Ns_1, erru_1, erreps_1)
  convergence_plot(Ns_1, erru2_1, erreps2_1)

# experiment 2
u_2(x) = sin(2*π*x[1]*x[2])
erru_2, erreps_2, erru2_2, erreps2_2, Ns_2 = h_refinement(u_2,ncells_1)  
  
  convergence_plot(Ns_2, erru_2, erreps_2)
  convergence_plot(Ns_2, erru2_2, erreps2_2)
  
# experiment 3
u_3(x) = (x[1]^7-x[1])*(x[2]^5-x[2])
erru_3, erreps_3, Ns_3 = h_refinement(u_3,ncells_1)  
  
convergence_plot(Ns_3, erru_3, erreps_3)



function slope(hs,errors)
  x = log.(hs)
  y = log.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

slope(Ns_1,erru_1)

