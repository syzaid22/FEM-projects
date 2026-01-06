# poromechanics second go, from ElasticityMixedTensor
module PoroelasticityMixedTensor_mixedBCTests
  using Gridap
  #using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  push!(LOAD_PATH, joinpath(@__DIR__, "Poroelasticity/src"))
  using GridapMixedViscoelasticityReactionDiffusion

  # Material parameters
    # const E = 1.0e2
    # const ν = 0.49
    # const λ = (E*ν)/((1+ν)*(1-2*ν))
    # const μ = E/(2*(1+ν))
    K = TensorValue(1.0, 0.0, 0.0, 1.0) # this will be variable later...
    Kinv = TensorValue(1.0, 0.0, 0.0, 1.0) # and so will this.
    const s_0 = 1
    const α = 1
    const λ = 1
    const μ = 1
    const d = 2

  calC(τ) = 2*μ*τ + λ*tr(τ)*one(τ)

  uex(x) = VectorValue(-cos(x[2])*x[1],x[2]*sin(x[1])) # VectorValue(0.1*cos(π*x[1])*sin(π*x[2])+0.15/λ*x[1]^2,
                     # -0.1*sin(π*x[1])*cos(π*x[2])+0.15/λ*x[2]^2)
  pex(x) = cos(x[1])*x[2]^2
  zex(x) = -(K⋅∇(pex)(x))
  σex(x) = (calC∘ε(uex))(x) - α*pex(x)*I
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))

  fex(x) = -(∇⋅σex)(x)
  gex(x) = s_0*pex(x) + α*(∇⋅uex)(x) + (∇⋅zex)(x) #

  comp1=extract_component(1)
  comp2=extract_component(2)
  row1=extract_row2d(1)
  row2=extract_row2d(2)

  function solve_poroelasticity(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

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

  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)
  Vh_ = TestFESpace(model,reffe_u,conformity=:L2)

  Sh1 = TrialFESpace(Sh_,VectorValue(0.0,0.0))
  Sh2 = TrialFESpace(Sh_,VectorValue(0.0,0.0))
  Ph = TrialFESpace(Gh_,pex)
  Vh = TrialFESpace(Vh_,uex)
  Gh = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,VectorValue(0.0,0.0))

  Y = MultiFieldFESpace([Sh_,Sh_,Gh_,Vh_,Gh_,Sh_])
  X = MultiFieldFESpace([Sh1,Sh2,Ph,Vh,Gh,Zh])

  a1((σ1,σ2),(τ1,τ2)) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2))dΩ -
                        ∫(λ/(2*μ*(2*μ+ d*λ))*(comp1∘σ1+comp2∘σ2)*(comp1∘τ1+comp2∘τ2))dΩ # C^{-1}σ:τ
  a2(q,(τ1,τ2)) = ∫( (α/(2*μ+ d*λ))*(q*(comp1∘τ1+comp2∘τ2)))dΩ
  a3(p,q) = ∫((s_0 + d*α^2/(2*μ + d*λ))*(p*q))dΩ

  a((σ1,σ2,p),(τ1,τ2,q)) =  a1((σ1,σ2),(τ1,τ2)) + a2(p,(τ1,τ2)) + a2(q,(σ1,σ2)) +  a3(p,q) 
 
  b1((τ1,τ2),(v,η)) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2) + η*(comp2∘τ1-comp1∘τ2))dΩ
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 

  b((τ1,τ2,q),(v,η,w)) =  b1((τ1,τ2),(v,η)) + b2(q,w)

  c(z,w) = ∫((Kinv⋅z)⋅w)dΩ

  F(τ1,τ2,q) =  ∫((τ1⋅n_ΓD)*(comp1∘uex) + (τ2⋅n_ΓD)*(comp2∘uex))dΓD + ∫(gex*q)dΩ 
  G(v,w) = ∫(pex*(w⋅n_ΓD))dΓD - ∫(fex⋅v)dΩ 

  lhs((σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) =  a((σ1,σ2,p),(τ1,τ2,q)) + b((τ1,τ2,q),(u,γ,z)) + b((σ1,σ2,p),(v,η,w)) - c(z,w)
  rhs((τ1,τ2,q,v,η,w)) =  F(τ1,τ2,q) + G(v,w)

  op = AffineFEOperator(lhs,rhs,X,Y)
  σh1, σh2, ph, uh, γh, zh = solve(op)

  mkpath("poroelasticity_output")
  if generate_output
      writevtk(Ω,"poroelasticity_output/convergence_AFW=$(num_cells(model))",order=1,
            cellfields=["σ1"=>σh1,"σ2"=>σh2, "p"=>ph,  "u"=>uh, "γ"=>γh, "z"=>zh])
      writevtk(model,"poroelasticity_output/model")
  end

  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  eph = pex-ph
  euh  = uex-uh
  eγh  = comp2∘row1∘γex-γh
  ezh = zex-zh

  error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h))dΩ))
  error_p = sqrt(sum(∫(eph*eph)dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγh*eγh)dΩ))
  error_z = sqrt(sum(∫(ezh⋅ezh)dΩ + ∫((∇⋅ezh)*(∇⋅ezh))dΩ))

  error_σ,error_p,error_u, error_γ, error_z, Gridap.FESpaces.num_free_dofs(X)
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
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_unit_square!(model)
      
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

    println("========================================================================")
    println("   DoF  &    h   &  e(p)   &  r(p)  &  e(z)   &  r(z)                   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], ep[nk], rp[nk], ez[nk], rz[nk]);
    end

    println("========================================================================")
  end
  convergence_test(;nkmax=6,k=0,generate_output=true)
end



