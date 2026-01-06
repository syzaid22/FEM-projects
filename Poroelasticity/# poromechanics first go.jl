# poromechanics first go
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.Fields
using LinearAlgebra:tr


"define these as functions of x?"
function row1(σ::TensorValue)
  return VectorValue(σ[1,1],σ[1,2]) 
end

function row2(σ::TensorValue)
  return VectorValue(σ[2,1],σ[2,2])
end

function cmbn(σ1::VectorValue,σ2::VectorValue) # combines rows back together into a matrix
 return TensorValue(σ1[1],σ1[2],σ2[1],σ2[2])
end

function Cinv(σ::TensorValue) # describes the action of the compliance tensor
  return (0.5/μ)*(σ-(λ/(2*μ+d*λ))*tr(σ)*one(σ))
end

function poromechanics(p,u1,u2,N) # take the components of u separately 
# setting up parameter values
Kinv = TensorValue(0.25,0.0,0.0,0.04) # this will be variable later
s_0 = 1
α = 1
λ = 1
μ = 1

u(x) = VectorValue(u1(x),u2(x))

"should we define σ row-wise from the start? "

γ(x) = skew_symmetric_gradient(u)(x)
σ(x) = symmetric_gradient(u)(x)

f(x) = -(∇⋅σ)(x)
g(x) = s_0*p(x) + α*(∇⋅u)(x) + (∇⋅z)(x) # used a mix of the strong form equations

domain = (0,1,0,1)
partition = (N,N)
d = length(partition) # dimension of the physical space

model = CartesianDiscreteModel(domain, partition)
order = 1

RefLag = ReferenceFE(lagrangian,Float64,order)
RefBDM = ReferenceFE(bdm,Float64,order)

# construct test spaces for each row of the matrix
Hdiv_tens1 = TestFESpace(model,RefBDM,
            conformity=:HDiv,dirichlet_tags=[5,6])
Hdiv_tens2 = TestFESpace(model,RefBDM,
            conformity=:HDiv,dirichlet_tags=[5,6])

L2 = TestFESpace(model,RefLag,conformity=:L2)    

L2_vec = TestFESpace(model,RefLag,conformity=:L2) # do i need to treat this differently to L2

# note L2_skew, in two dimensions, only has one degree of freedom
L2_skew = TestFESpace(model,RefLag,conformity=:L2)

Hdiv = TestFESpace(model,RefBDM,
              conformity=:HDiv,dirichlet_tags=[5,6])

T1a = TrialFESpace(Hdiv_tens1,row1∘σ)
T1b = TrialFESpace(Hdiv_tens2,row2∘σ)
T2 = TrialFESpace(L2,p)
T3 = TrialFESpace(L2_vec,u)
T4 = TrialFESpace(L2_skew)
T5 = TrialFESpace(Hdiv_vec,z)

trials = MultiFieldFESpace([Hdiv_tens1,Hdiv_tens2,L2,L2_vec,L2_skew,Hdiv])
tests = MultiFieldFESpace([T1a,T1b,T2,T3,T4,T5])

tri = Triangulation(model)
degree = 8
dΩ = Measure(tri,degree)

btri = BoundaryTriangulation(model,tags=[7,8]) # on Γ_D
dΓ = Measure(btri,degree)

nb = get_normal_vector(btri)

cnst = α/(2*μ+d*λ) 

a((σ1,σ2,p),(τ1,τ2,q)) = ∫( Cinv(cmbn(σ1,σ2)) ⊙ cmbn(τ1,τ2) )dΩ + 
∫( (cnst*p)*(tr(cmbn(τ1,τ2))) )dΩ + ∫( (cnst*q)*(tr(cmbn(σ1,σ2))) )dΩ + (s_0+d*α*cnst)*∫(p*q)dΩ 

b((τ1,τ2,q),(v,η,w)) = ∫( v⋅(VectorValue(∇⋅τ1,∇⋅τ2)) )dΩ + ∫( η*(τ1[2]-τ2[1]) )dΩ + ∫( q*∇⋅w )dΩ 

c((u,γ,z),(v,η,w)) = ∫((Kinv*z)⋅w)dΩ

F((τ1,τ2,q))=∫(g*q)dΩ + ∫(u⋅VectorValue(τ1⋅nb,τ2⋅nb))dΓ
G((v,η,w))= -∫(f⋅v)dΩ + ∫((w⋅nb)*p)dΓ 

A((σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) = a((σ1,σ2,p),(τ1,τ2,q)) + b((τ1,τ2,q),(u,γ,z)) 
+ b((σ1,σ2,p),(v,η,w)) - c((u,γ,z),(v,η,w))

H((τ1,τ2,q,v,η,w)) = F((τ1,τ2,q)) + G((v,η,w))   

op = AffineFEOperator(A,H,trials,tests)
xh = solve(op)
σ1h,σ2h,ph,uh,γh,zh = xh

# L2 inner product
e(w,v) = ∫( w⋅v )dΩ
# our inner product on H(div)
d(w,v) = e(w,v) + ∫((∇⋅w)*(∇⋅v))dΩ


for i in [σ1,σ2,p,u,γ,z]

  error_$i = 2 

end

eu = u - uh
ep = p - ph

  erru = sqrt(sum(c(eu,eu)))

  errp = sqrt(sum(d(ep,ep)))


return erru, errepsu, errp, errepsp

end #function


u(x) = VectorValue(x[1]*x[2],x[1]*x[2])

