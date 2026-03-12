#3D poromechanics
module Poroelasticity3D
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  push!(LOAD_PATH, joinpath(@__DIR__,"src"))
  using GridapMixedViscoelasticityReactionDiffusion

  # Material parameters
    # const E = 16000
    # const ν = 0.479
    # const λ = (E*ν)/((1+ν)*(1-2*ν))
    # const μ = E/(2*(1+ν))
    const λ = 1
    const μ = 1

    K_component = 1
    K = TensorValue(K_component, 0.0, 0.0, 0.0, K_component, 0.0, 0.0, 0.0, K_component) # this will be variable later...
    Kinv = TensorValue(1/K_component, 0.0, 0.0, 0.0, 1/K_component, 0.0, 0.0, 0.0, 1/K_component) # and so will this.
    const s_0 = 1#e-3
    const α = 1

    print("λ   = $(λ)\n")
    print("μ   = $(μ)\n")
    print("s_0 = $(s_0)\n")
    print("K   = $(K_component)\n")

    const d = 3 # dimension of spatial domain

  calC(τ) = 2*μ*τ + λ*tr(τ)*one(τ)

  uex(x) = VectorValue(-cos(x[2])*x[1],x[2]*sin(x[1]),-(x[2]^2)*sin(x[1])) # VectorValue(0.1*cos(π*x[1])*sin(π*x[2])+0.15/λ*x[1]^2,
                     # -0.1*sin(π*x[1])*cos(π*x[2])+0.15/λ*x[2]^2)
  pex(x) = cos(x[1])*x[2]^2
  zex(x) = -(K⋅∇(pex)(x))
  σex(x) = (calC∘ε(uex))(x) - α*pex(x)*(one∘ε(uex))(x)
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))

  fex(x) = -(∇⋅σex)(x)   
  gex(x) = s_0*pex(x) + α*(∇⋅uex)(x) + (∇⋅zex)(x) 

  comp1=extract_component(1)
  comp2=extract_component(2)
  comp3=extract_component(3)
  row1=extract_row3d(1)
  row2=extract_row3d(2)
  row3=extract_row3d(3)

  function solve_poroelasticity(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ_ = ReferenceFE(bdm,Float64,k+1)
  reffe_u_ = ReferenceFE(lagrangian,VectorValue{3,Float64},k)
  reffe_γ_ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 5+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  #ΓN = BoundaryTriangulation(model,tags = "Gamma_sig")
  ΓD = BoundaryTriangulation(model,tags = "Gamma_u")
  #n_ΓN = get_normal_vector(ΓN) 
  n_ΓD = get_normal_vector(ΓD)
  #dΓN = Measure(ΓN,degree)
  dΓD = Measure(ΓD,degree)

  Sh_ = TestFESpace(model,reffe_σ_,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Gh_ = TestFESpace(model,reffe_γ_,conformity=:L2)
  Vh_ = TestFESpace(model,reffe_u_,conformity=:L2)

  Sh1 = TrialFESpace(Sh_,row1∘σex) 
  Sh2 = TrialFESpace(Sh_,row2∘σex)
  Sh3 = TrialFESpace(Sh_,row3∘σex)
  Ph = TrialFESpace(Gh_)
  Vh = TrialFESpace(Vh_)
  Gh1 = TrialFESpace(Gh_)
  Gh2 = TrialFESpace(Gh_)
  Gh3 = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,zex)

  Y = MultiFieldFESpace([Sh_,Sh_,Sh_,Gh_,Vh_,Gh_,Gh_,Gh_,Sh_])
  X = MultiFieldFESpace([Sh1,Sh2,Sh3,Ph,Vh,Gh1,Gh2,Gh3,Zh])

  a1((σ1,σ2,σ3),(τ1,τ2,τ3)) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2 + σ3⋅τ3))dΩ -
                        ∫(λ/(2*μ*(2*μ+ d*λ))*(comp1∘σ1+comp2∘σ2+comp3∘σ3)*(comp1∘τ1+comp2∘τ2+comp3∘τ3))dΩ # C^{-1}σ:τ
  a2(q,(τ1,τ2,τ3)) = ∫( (α/(2*μ+ d*λ))*(q*(comp1∘τ1+comp2∘τ2+comp3∘τ3)))dΩ
  a3(p,q) = ∫((s_0 + d*α^2/(2*μ + d*λ))*(p*q))dΩ

  a((σ1,σ2,σ3,p),(τ1,τ2,τ3,q)) =  a1((σ1,σ2,σ3),(τ1,τ2,τ3)) + a2(p,(τ1,τ2,τ3)) + a2(q,(σ1,σ2,σ3)) +  a3(p,q) 
 
  b1((τ1,τ2,τ3),(v,η1,η2,η3)) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2)+(comp3∘v)*(∇⋅τ3))dΩ + ∫(η1*(comp2∘τ1-comp1∘τ2) + η2*(comp3∘τ1-comp1∘τ3) + η3*(comp2∘τ3-comp3∘τ2))dΩ 
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 

  b((τ1,τ2,τ3,q),(v,η1,η2,η3,w)) =  b1((τ1,τ2,τ3),(v,η1,η2,η3)) + b2(q,w)

  c(z,w) = ∫((Kinv⋅z)⋅w)dΩ 

  F(τ1,τ2,τ3,q) =  ∫((τ1⋅n_ΓD)*(comp1∘uex) + (τ2⋅n_ΓD)*(comp2∘uex) + (τ3⋅n_ΓD)*(comp3∘uex))dΓD + ∫(gex*q)dΩ 
  G(v,w) = ∫(pex*(w⋅n_ΓD))dΓD - ∫(fex⋅v)dΩ 

  lhs((σ1,σ2,σ3,p,u,γ1,γ2,γ3,z),(τ1,τ2,τ3,q,v,η1,η2,η3,w)) = a((σ1,σ2,σ3,p),(τ1,τ2,τ3,q)) + b((τ1,τ2,τ3,q),(u,γ1,γ2,γ3,z)) + b((σ1,σ2,σ3,p),(v,η1,η2,η3,w)) - c(z,w)
  rhs((τ1,τ2,τ3,q,v,η1,η2,η3,w)) =  F(τ1,τ2,τ3,q) + G(v,w)                                                        

  op = AffineFEOperator(lhs,rhs,X,Y)
  σh1, σh2, σh3, ph, uh, γh1, γh2, γh3, zh  = solve(op)

  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  eσ3h = (row3∘σex)-σh3
  eph = pex-ph
  euh  = uex-uh
  eγ1h  = comp1∘row2∘γex-γh1
  eγ2h  = comp1∘row3∘γex-γh2
  eγ3h  = comp3∘row2∘γex-γh3
  ezh = zex-zh

  error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h+eσ3h⋅eσ3h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h)+(∇⋅eσ3h)*(∇⋅eσ3h))dΩ))
  error_p = sqrt(sum(∫(eph*eph)dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγ1h*eγ1h + eγ2h*eγ2h + eγ3h*eγ3h)dΩ))
  error_z = sqrt(sum(∫(ezh⋅ezh)dΩ + ∫((∇⋅ezh)*(∇⋅ezh))dΩ))

  error_σ,error_p,error_u,error_γ,error_z, Gridap.FESpaces.num_free_dofs(X)
  end



  function  convergence_test(; nkmax, k=0, generate_output=false)
    eσ   = Float64[]
    rσ   = Float64[]
    eu   = Float64[]
    ru   = Float64[]
    eγ   = Float64[]
    rγ   = Float64[]
    ep   = Float64[]
    rp   = Float64[] 
    ez   = Float64[]
    rz   = Float64[]

    push!(ru,0.)
    push!(rσ,0.)
    push!(rγ,0.)
    push!(rp,0.)
    push!(rz,0.)

    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_cube(nk) # Discrete model
       setup_model_labels_unit_cube!(model)
      
       error_σ,error_p,error_u, error_γ, error_z, ndofs=solve_poroelasticity(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk) # i.e. using diameter of simplices

       push!(eσ,error_σ)
       push!(eu,error_u)
       push!(eγ,error_γ)
       push!(ep,error_p)
       push!(ez,error_z)

       if nk>1
         push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rγ, log(eγ[nk]/eγ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rz, log(ez[nk]/ez[nk-1])/log(hh[nk]/hh[nk-1]))

       end
    end

    println("========================================================================")
    println("   DoF  &    h   &  e(σ)   &  r(σ)  &  e(u)   &  r(u)  & e(γ)  & r(γ)   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], eσ[nk], rσ[nk], eu[nk], ru[nk], eγ[nk], rγ[nk]);
    end

    println("========================================================================")
    println("   DoF  &    h   &  e(p)   &  r(p)  &  e(z)   &  r(z)                   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], ep[nk], rp[nk], ez[nk], rz[nk]);
    end

    println("========================================================================")
  end
  convergence_test(;nkmax=3,k=0,generate_output=false)
end

