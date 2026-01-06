# minres on darcy

using Gridap
using Plots
using DataFrames, CSV

# i.e. u is vector-valued, p is scalar-valued
function minres_darcy(u,p,N)

Kinv = TensorValue(0.25,0.0,0.0,0.04)

f(x) = -(∇⋅u)(x)
g(x) = Kinv⋅u(x) + ∇(p)(x)


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

a(w,v) = ∫((Kinv⋅w)⋅v)dΩ

b(ρ,v)= ∫(-ρ⋅(∇⋅(v)))dΩ

# our inner product on H(div)
c(w,v) = a(w,v) + ∫((∇⋅(w))*(∇⋅(v)))dΩ

# L2 inner product
d(w,v) = ∫( w⋅v )dΩ

F(q)=∫(f*q)dΩ
G(v)= ∫(g⋅v)dΩ + ∫(-(v⋅nb)*p)dΓ  # edit 2: added boundary term 

A((ϵu,ϵp,w,ρ),(v,q,tu,tp))=c(ϵu,v)+a(w,v)+b(ρ,v) + d(ϵp,q)+b(q,w)+ a(tu,ϵu)+b(ϵp,tu) + b(tp,ϵu)
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

return erru, errepsu, errp, errepsp

end #function



function h_refinement(u,p,ncells)

  size = length(ncells)

  erru_h = Array{Float64}(undef,size)
  errepsu_h = Array{Float64}(undef,size)
  
  errp_h = Array{Float64}(undef,size)
  errepsp_h = Array{Float64}(undef,size)

    for i = 1:size
      erru, errepsu, errp, errepsp = minres_darcy(u,p,ncells[i])

      erru_h[i] = erru
      errepsu_h[i] = errepsu

      errp_h[i] = errp
      errepsp_h[i] = errepsp
    end

  return erru_h, errepsu_h, errp_h, errepsp_h

end


function convergence_plot(Narr,errors1,errors2)
  plot(Narr,errors1,label=["u-uh"],shape=:auto)
  plot!(xaxis=:log, yaxis=:log,
  shape=:auto,
  label=["errors"],
  xlabel="h",ylabel="error norm")
  plot!(Narr,errors2,label=["ϵh"],shape=:auto)
end

# first experiment!
	u_1(x) = -π*VectorValue(sin(π*x[2])*cos(π*x[1]),cos(π*x[2])*sin(π*x[1]))
  p_1(x) = sin(π*x[1])*sin(π*x[2])
	ncells_1 = [ 2^i for i in 2:5 ]
  h_1 = 1 ./ ncells_1
	erru_1, errepsu_1, errp_1, errepsp_1 = h_refinement(u_1,p_1,ncells_1)

  convergence_plot(h_1, erru_1, errepsu_1)
  convergence_plot(h_1, errp_1, errepsp_1)

# 2nd experiment
	u_2(x) = VectorValue(-cos(x[2])*x[1],x[2]*sin(x[1]))
  p_2(x) = cos(x[1])*x[2]^2
	ncells_2 = [ 2^i for i in 2:5 ]
  h_2 = 1 ./ ncells_2
	erru_2, errepsu_2, errp_2, errepsp_2 = h_refinement(u_2,p_2,ncells_2)

  convergence_plot(h_2, erru_2, errepsu_2)
  convergence_plot(h_2, errp_2, errepsp_2)

  df1 = DataFrame(size = h_2, error = erru_2)
  CSV.write("erru1.dat", df1)
  df2 = DataFrame(size = h_2, error = errepsu_2)
  CSV.write("erru2.dat", df2)

  df3 = DataFrame(size = h_2, error = errp_2)
  CSV.write("errp.dat", df3)
  df4 = DataFrame(size = h_2, error = errepsp_2)
  CSV.write("errepsp.dat", df4)

 
