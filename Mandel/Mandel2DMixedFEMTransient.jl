
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  const E = 1.0E3
  const ν = 1.0/3.0
  const λ = (E*ν)/((1+ν)*(1-2*ν))
  const μ = E/(2*(1+ν))
  const κ = 5.1E-8
  const μf = 1.0E-3
  const s_0 = 4.0E-10
  const α = 0.9
  const ρ = 1.0

  f  = VectorValue(0.0,0.0)
  mp = 0.0
  traction(t) = x -> VectorValue(0.0,-1.0E2)
  zerov = VectorValue(0.0,0.0)
  zerovt(t) = x -> VectorValue(0.0,0.0)
  p_right = 0.0
  zerop = 0.0

  k   = 1  

  tang_left = VectorValue(0.0,-1.0)
  tang_bot = VectorValue(1.0,0.0)
 

  print("λ   = $(λ)\n")
  print("μ   = $(μ)\n")


  model = CartesianDiscreteModel((0,1,0,1), (30,30)) |> simplexify

  labels = get_face_labeling(model)
  add_tag!(labels,"Gamma_right",[8]) #right 
  add_tag!(labels,"Gamma_left",[7]) # left
  add_tag!(labels,"Gamma_top",[6]) # top
  add_tag!(labels,"Gamma_bot",[5]) # bottom
 

  function extract_component(component)
    return x -> x[component]
  end

  function extract_row2d(row)
    return x -> VectorValue(x[row,1],x[row,2]) 
  end

  comp1=extract_component(1)
  comp2=extract_component(2)
  row1=extract_row2d(1)
  row2=extract_row2d(2)

  # Reference FEs
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)
  
  # Numerical integration
  degree = 4+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  Γleft = BoundaryTriangulation(model,tags = "Gamma_left")
  n_Γleft = get_normal_vector(Γleft)
  dΓleft = Measure(Γleft  ,degree)
  Γbot = BoundaryTriangulation(model,tags = "Gamma_bot")
  n_Γbot = get_normal_vector(Γbot)
  dΓbot = Measure(Γbot  ,degree)
  Γright = BoundaryTriangulation(model,tags = "Gamma_right")
  n_Γright = get_normal_vector(Γright)
  dΓright = Measure(Γright  ,degree)  


  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags=["Gamma_top","Gamma_right"],conformity=:HDiv)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)
  Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
  Zh_ = TestFESpace(model,reffe_σ,dirichlet_tags=["Gamma_bot","Gamma_left","Gamma_top"],conformity=:HDiv) 
  

  # need to check TOP: want that sigma*n = traction on the top boundary, and sigma*n = 0 on right; here we can prescribe and vector such its second component is zero
  Sh1 = TransientTrialFESpace(Sh_,[zerovt,zerovt]) # need that sigma12 is zero on the top boundary, here we can prescribe and vector such its second component is zero  
  Sh2 = TransientTrialFESpace(Sh_,[traction,zerovt]) # need that sigma22 is equal to the traction on the top boundary, here we can prescribe and vector such its first component is zero 
  Ph = TransientTrialFESpace(Gh_)
  Vh = TrialFESpace(Vh_)
  Gh = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,zerov) 
  Y = MultiFieldFESpace([Sh_,Sh_,Gh_,Vh_,Gh_,Sh_])
  X = MultiFieldFESpace([Sh1,Sh2,Ph,Vh,Gh,Zh])


  a1(t,(σ1,σ2),(τ1,τ2)) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2))dΩ -
                        ∫(λ/(2*μ*(2*μ+ 2*λ))*(comp1∘σ1+comp2∘σ2)*(comp1∘τ1+comp2∘τ2))dΩ # C^{-1}σ:τ
  a2(t,q,(τ1,τ2)) = ∫( (α/(2*μ+ 2*λ))*(q*(comp1∘τ1+comp2∘τ2)))dΩ
  a3(t,p,q) = ∫((s_0 + 2*α^2/(2*μ + 2*λ))*(p*q))dΩ

  b1(t,(τ1,τ2),(v,η)) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2) + η*(comp2∘τ1-comp1∘τ2))dΩ 
  b2(t,q,w) = ∫( q*(∇⋅w) )dΩ 
  b(t,(τ1,τ2,q),(v,η,w)) =  b1(t,(τ1,τ2),(v,η)) + b2(t,q,w)

  c(t,z,w) = ∫((κ^(-1)*z)⋅w)dΩ 

  # postulate: <tau*n,u> = <tau*n . n , u.n> + <(tau*n).t, u.t>
  # as we are imposing u.n = 0 (naturally) then only the other term remains 

  aF(t,τ1,τ2,u) = ∫((-1)*(τ1⋅n_Γleft)*tang_left[1]*(u⋅tang_left)+(-1)*(τ2⋅n_Γleft)*tang_left[2]*(u⋅tang_left))dΓleft + 
                  ∫((-1)*(τ1⋅n_Γbot)*tang_bot[1]*(u⋅tang_bot)+(-1)*(τ2⋅n_Γbot)*tang_bot[2]*(u⋅tang_bot))dΓbot 

  G(t,q,v,w) = ∫((-1)*p_right*(w⋅n_Γright))dΓright - ∫(ρ*(f⋅v))dΩ - ∫(mp*q)dΩ

  stiffness(t,(σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) = a1(t,(σ1,σ2),(τ1,τ2)) + a2(t,p,(τ1,τ2)) +
                                                 b(t,(τ1,τ2,q),(u,γ,z)) + 
                                                 b(t,(σ1,σ2,p),(v,η,w)) - c(t, z,w) + aF(t,τ1,τ2,u)
  mass(t,(σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) = a2(t,q,(σ1,σ2)) +  a3(t,p,q)
  rhs(t,(τ1,τ2,q,v,η,w)) =  G(t,q,v,w)

  op = TransientLinearFEOperator((stiffness,mass),rhs,X,Y, constant_forms=(true, true))
  ls = LUSolver()

  θ = 1.0; Δt = 0.1
  tsolver = ThetaMethod(ls, Δt, θ)
  
  t0 = 0.0; tF = 1.0
  zerof = 0.0
  # initial conditions 
  x0 = interpolate_everywhere([zerov,zerov,zerof,zerov,zerof,zerov],X(t0))

  fesltn = solve(tsolver, op, t0, tF, x0)
  allsol = collect(fesltn)  # collect all time steps
  tfinal, xfinal_state = allsol[end]  # get final time and solution

  σ1h = xfinal_state[1]
  σ2h = xfinal_state[2]
  ph = xfinal_state[3]
  uh = xfinal_state[4]
  γh = xfinal_state[5]
  zh = xfinal_state[6]

  writevtk(Ω,"Mandel/outputs/Mandel2D_AFW_transient",order=1, cellfields=["σ1"=>σ1h,"σ2"=>σ2h, "p"=>ph,  "u"=>uh, "γ"=>γh, "z"=>zh])
  