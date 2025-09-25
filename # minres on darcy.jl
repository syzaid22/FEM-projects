# minres on darcy

using Gridap
using Plots

function minres_darcy(u,p,N)

Kinv = TensorValue(0.25,0.0,0.0,0.04)

f(x) = 2*(π^2)*sin(π*x[1])*sin(π*x[2])
g(x) = u(x) + ∇(p)(x)


domain = (0,1,0,1)
partition = (N,N)
model = CartesianDiscreteModel(domain, partition)

order2=2
order1=1

V_rt = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,order2),
            conformity=:HDiv,dirichlet_tags=[5,6])
            
V_L2 = TestFESpace(model,ReferenceFE(lagrangian,Float64,order2),
            conformity=:L2)    

U_rt = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,order1),
            conformity=:HDiv,dirichlet_tags=[5,6])

U_L2 = TestFESpace(model,ReferenceFE(lagrangian,Float64,order1),
            conformity=:L2)

Vt_rt = TrialFESpace(V_rt,VectorValue(0.0,0.0))
Vt_L2 = TrialFESpace(V_L2)
Ut_rt = TrialFESpace(U_rt,u)   # edit 1: added possibly non-zero BC
Ut_L2 = TrialFESpace(U_L2)

trials = MultiFieldFESpace([Vt_rt,Vt_L2,Ut_rt,Ut_L2])
tests = MultiFieldFESpace([V_rt,V_L2,U_rt,U_L2])

tri = Triangulation(model)
degree = 8
dΩ = Measure(tri,degree)

btri = BoundaryTriangulation(model,tags=[7,8])
dΓ = Measure(btri,degree)

nb = get_normal_vector(btri)

# using Kinvhalf = Kinv^0.5
Kinvhalf = TensorValue(1.0,0.0,0.0,1.0)

a(w,v) = ∫((Kinvhalf⋅w)⋅(Kinvhalf⋅v))dΩ

b(ρ,v)= ∫(-ρ⋅(∇⋅(v)))dΩ

# our inner product on H(div)
c(w,v) = a(w,v) + ∫((∇⋅(w))*(∇⋅(v)))dΩ

# L2 inner product
d(w,v) = ∫( w⋅v )dΩ

F(q)=∫(f*q)dΩ
G(v)= ∫(g⋅v)dΩ + ∫(-(v⋅nb)*p)dΓ  # edit 2: added boundary term #

A((ϵu,ϵp,w,ρ),(v,q,tu,tp))=c(ϵu,v)+a(w,v)+b(ρ,v) + d(ϵp,q)-b(q,w)+ a(tu,ϵu)+b(ϵp,tu) - b(tp,ϵu)
H((v,q,tu,tp)) = F(q) + G(v)   # edit 3: corrected order of epsilons vs solutions in the smaller spaces

op = AffineFEOperator(A,H,trials,tests)
xh = solve(op)
ϵuh,ϵph,uh,ph = xh

eu = u - uh
ep = p - ph



  erru = sqrt(sum(c(eu,eu)))
  errepsu = sqrt(sum(c(ϵuh,ϵuh)))

  errp = sqrt(sum(d(ep,ep)))
  errepsp = sqrt(sum(d(ϵph,ϵph)))


writevtk(tri,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph])

return erru, errepsu, errp, errepsp

end #function





function h_refinement(u,p,ncells)

  erru_h = Float64[]
  errepsu_h = Float64[]
  
  errp_h = Float64[]
  errepsp_h = Float64[]

  Ns = Int[]

    for N in ncells

      erru, errepsu, errp, errepsp = minres_darcy(u,p,N)
      push!(erru_h,erru)
      push!(errepsu_h,errepsu)

      push!(errp_h,errp)
      push!(errepsp_h,errepsp)

      push!(Ns,N)

    end

  return erru_h, errepsu_h, errp_h, errepsp_h, Ns

end


function convergence_plot(Narr,errors1,errors2)
  plot(Narr,errors1,label=["u-uh"],shape=:auto)
  plot!(xaxis=:log, yaxis=:log,
  shape=:auto,
  label=["errors"],
  xlabel="N",ylabel="error norm")
  plot!(Narr,errors2,label=["ϵh"],shape=:auto)
end

# first experiment!
	u_1(x) = -π*VectorValue(sin(π*x[2])*cos(π*x[1]),cos(π*x[2])*sin(π*x[1]))
  p_1(x) = sin(π*x[1])*sin(π*x[2])
	ncells_1 = [ 2^i for i in 2:5 ]
	erru_1, errepsu_1, errp_1, errepsp_1,  Ns_1 = h_refinement(u_1,p_1,ncells_1)

  convergence_plot(Ns_1, erru_1, errepsu_1)
  convergence_plot(Ns_1, errp_1, errepsp_1)

  convergence_plot(Ns_1, erru_1, errp_1)


  writevtk(tri,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph])
