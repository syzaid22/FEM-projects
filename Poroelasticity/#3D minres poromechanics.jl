# 3D minres poromechanics
# needs some work. γ still doesn't converge.
module PoroelasticityMinRes3D
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  push!(LOAD_PATH, joinpath(@__DIR__, "src"))
  using GridapMixedViscoelasticityReactionDiffusion

  # Material parameters
    # const E = 16000
    # const ν = 0.479
    # const λ = (E*ν)/((1+ν)*(1-2*ν))
    # const μ = E/(2*(1+ν))
    const λ = 1
    const μ = 1

    K_component = 1.0
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

  function solve_minres_poroelasticity(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ_ = ReferenceFE(bdm,Float64,k+1)
  reffe_u_ = ReferenceFE(lagrangian,VectorValue{3,Float64},k)
  reffe_γ_ = ReferenceFE(lagrangian,Float64,k)

  reffe_σ_1 = ReferenceFE(bdm,Float64,k+2)
  reffe_u_1 = ReferenceFE(lagrangian,VectorValue{3,Float64},k+1)
  reffe_γ_1 = ReferenceFE(lagrangian,Float64,k+1)

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

  Sh_1 = TestFESpace(model,reffe_σ_1,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Gh_1 = TestFESpace(model,reffe_γ_1,conformity=:L2)
  Vh_1 = TestFESpace(model,reffe_u_1,conformity=:L2)

  Sh1 = TrialFESpace(Sh_,row1∘σex) 
  Sh2 = TrialFESpace(Sh_,row2∘σex)
  Sh3 = TrialFESpace(Sh_,row3∘σex)
  Ph = TrialFESpace(Gh_)
  Vh = TrialFESpace(Vh_)
  Gh1 = TrialFESpace(Gh_)
  Gh2 = TrialFESpace(Gh_)
  Gh3 = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,zex)

  rSh1 = TrialFESpace(Sh_1)#,VectorValue(0.0,0.0))
  rSh2 = TrialFESpace(Sh_1)#,VectorValue(0.0,0.0))
  rSh3 = TrialFESpace(Sh_1)#,VectorValue(0.0,0.0))
  rPh = TrialFESpace(Gh_1)
  rVh = TrialFESpace(Vh_1)
  rGh1 = TrialFESpace(Gh_1)
  rGh2 = TrialFESpace(Gh_1)
  rGh3 = TrialFESpace(Gh_1)
  rZh = TrialFESpace(Sh_1)#,VectorValue(0.0,0.0))
  # what would the boundary conditions be on the test spaces; zero or none?

  Y = MultiFieldFESpace([Sh_1,Sh_1,Sh_1,Gh_1,Vh_1,Gh_1,Gh_1,Gh_1,Sh_1,Sh_,Sh_,Sh_,Gh_,Vh_,Gh_,Gh_,Gh_,Sh_])
  X = MultiFieldFESpace([rSh1,rSh2,rSh3,rPh,rVh,rGh1,rGh2,rGh3,rZh,Sh1,Sh2,Sh3,Ph,Vh,Gh1,Gh2,Gh3,Zh])

  a1((σ1,σ2,σ3),(τ1,τ2,τ3)) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2 + σ3⋅τ3))dΩ -
                        ∫(λ/(2*μ*(2*μ+ d*λ))*(comp1∘σ1+comp2∘σ2+comp3∘σ3)*(comp1∘τ1+comp2∘τ2+comp3∘τ3))dΩ # C^{-1}σ:τ
  a2(q,(τ1,τ2,τ3)) = ∫( (α/(2*μ+ d*λ))*(q*(comp1∘τ1+comp2∘τ2+comp3∘τ3)))dΩ
  a3(p,q) = ∫((s_0 + d*α^2/(2*μ + d*λ))*(p*q))dΩ

  a((σ1,σ2,σ3,p),(τ1,τ2,τ3,q)) =  a1((σ1,σ2,σ3),(τ1,τ2,τ3)) + a2(p,(τ1,τ2,τ3)) + a2(q,(σ1,σ2,σ3)) +  a3(p,q) 
 
  b1((τ1,τ2,τ3),(v,η1,η2,η3)) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2)+(comp3∘v)*(∇⋅τ3))dΩ + ∫(η1*(comp2∘τ1-comp1∘τ2) + η2*(comp3∘τ1-comp1∘τ3) + η3*(comp2∘τ3-comp3∘τ2))dΩ 
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 

  b((τ1,τ2,τ3,q),(v,η1,η2,η3,w)) =  b1((τ1,τ2,τ3),(v,η1,η2,η3)) + b2(q,w)

  c(z,w) = ∫((Kinv⋅z)⋅w)dΩ 

  d1(p,q) = ∫(p⋅q)dΩ # L2 inner product
  d2(z,w) = ∫(z⋅w + (∇⋅z)*(∇⋅w))dΩ # Hdiv inner product
  d3(z,w) = ∫((Kinv⋅z)⋅w + (∇⋅z)*(∇⋅w))dΩ # weighted Hdiv inner product

  F(τ1,τ2,τ3,q) =  ∫((τ1⋅n_ΓD)*(comp1∘uex) + (τ2⋅n_ΓD)*(comp2∘uex) + (τ3⋅n_ΓD)*(comp3∘uex))dΓD + ∫(gex*q)dΩ 
  G(v,w) = ∫(pex*(w⋅n_ΓD))dΓD - ∫(fex⋅v)dΩ 

  lhs((εσ1,εσ2,εσ3,εp,εu,εγ1,εγ2,εγ3,εz,σ1,σ2,σ3,p,u,γ1,γ2,γ3,z),(τ1,τ2,τ3,q,v,η1,η2,η3,w,φσ1,φσ2,φσ3,φp,φu,φγ1,φγ2,φγ3,φz)) =   a((σ1,σ2,σ3,p),(τ1,τ2,τ3,q)) + b((τ1,τ2,τ3,q),(u,γ1,γ2,γ3,z)) + b((σ1,σ2,σ3,p),(v,η1,η2,η3,w)) - c(z,w) + a((φσ1,φσ2,φσ3,φp),(εσ1,εσ2,εσ3,εp)) + b((φσ1,φσ2,φσ3,φp),(εu,εγ1,εγ2,εγ3,εz)) + b((εσ1,εσ2,εσ3,εp),(φu,φγ1,φγ2,φγ3,φz)) - c(φz,εz) + d2(εσ1,τ1) + d2(εσ2,τ2) + d2(εσ3,τ3) + ∫(εp*q)dΩ + ∫(εu⋅v)dΩ + ∫(2*εγ1*η1)dΩ + ∫(2*εγ2*η2)dΩ+ ∫(2*εγ3*η3)dΩ + d2(εz,w)
  rhs((τ1,τ2,τ3,q,v,η1,η2,η3,w,φσ1,φσ2,φσ3,φp,φu,φγ1,φγ2,φγ3,φz)) =  F(τ1,τ2,τ3,q) + G(v,w)                                                        

  op = AffineFEOperator(lhs,rhs,X,Y)
  εσh1, εσh2, εσh3, εph, εuh, εγh1, εγh2, εγh3, εzh, σh1, σh2, σh3, ph, uh, γh1, γh2, γh3, zh  = solve(op)

# this needs to be tweaked

#   mkpath("poroelasticityminres_output")
#   if generate_output
#       writevtk(Ω,"poroelasticityminres_output/convergence_AFW=$(num_cells(model))",order=1,
#             cellfields=["σ1"=>σh1,"σ2"=>σh2, "p"=>ph,  "u"=>uh, "γ"=>γh, "z"=>zh])
#       writevtk(model,"poroelasticityminres_output/model")
#   end

  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  eσ3h = (row3∘σex)-σh3
  eph = pex-ph
  euh  = uex-uh
  eγ1h  = comp2∘row1∘γex-γh1
  eγ2h  = comp1∘row3∘γex-γh2
  eγ3h  = comp3∘row2∘γex-γh3
  ezh = zex-zh

  error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h+eσ3h⋅eσ3h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h)+(∇⋅eσ3h)*(∇⋅eσ3h))dΩ))
  error_p = sqrt(sum(∫(eph*eph)dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγ1h*eγ1h + eγ2h*eγ2h + eγ3h*eγ3h)dΩ))
  error_z = sqrt(sum(d2(ezh,ezh)))

  size_εσh = sqrt(sum(∫(εσh1⋅εσh1+εσh2⋅εσh2+εσh3⋅εσh3)dΩ +
                     ∫((∇⋅εσh1)*(∇⋅εσh1)+(∇⋅εσh2)*(∇⋅εσh2)+(∇⋅εσh3)*(∇⋅εσh3))dΩ))
  size_εph = sqrt(sum(∫(εph*εph)dΩ))
  size_εuh = sqrt(sum(∫(εuh⋅εuh)dΩ))
  size_εγh = sqrt(sum(∫(εγh1*εγh1 + εγh2*εγh2 + εγh3*εγh3)dΩ))
  size_εzh = sqrt(sum(d2(εzh,εzh)))

  size_εσh,size_εph,size_εuh,size_εγh,size_εzh, error_σ,error_p,error_u,error_γ,error_z, Gridap.FESpaces.num_free_dofs(X)
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

    εσ   = Float64[]
    rεσ   = Float64[]
    εu   = Float64[]
    rεu   = Float64[]
    εγ   = Float64[]
    rεγ   = Float64[]
    εp   = Float64[]
    rεp   = Float64[] 
    εz   = Float64[]
    rεz   = Float64[]

    push!(ru,0.)
    push!(rσ,0.)
    push!(rγ,0.)
    push!(rp,0.)
    push!(rz,0.)

    push!(rεu,0.)
    push!(rεσ,0.)
    push!(rεγ,0.)
    push!(rεp,0.)
    push!(rεz,0.)

    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_cube(nk) # Discrete model
       setup_model_labels_unit_cube!(model)
      
       size_εσh,size_εph,size_εuh,size_εγh,size_εzh,error_σ,error_p,error_u, error_γ, error_z, ndofs=solve_minres_poroelasticity(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk) # i.e. using diameter of simplices

       push!(eσ,error_σ)
       push!(eu,error_u)
       push!(eγ,error_γ)
       push!(ep,error_p)
       push!(ez,error_z)

       push!(εσ,size_εσh)
       push!(εu,size_εuh)
       push!(εγ,size_εγh)
       push!(εp,size_εph)
       push!(εz,size_εzh)


       if nk>1
         push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rγ, log(eγ[nk]/eγ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rz, log(ez[nk]/ez[nk-1])/log(hh[nk]/hh[nk-1]))

         push!(rεσ, log(εσ[nk]/εσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rεu, log(εu[nk]/εu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rεγ, log(εγ[nk]/εγ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rεp, log(εp[nk]/εp[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rεz, log(εz[nk]/εz[nk-1])/log(hh[nk]/hh[nk-1]))
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
    println("   DoF  &    h   &  |εσ|   &  r(εσ)  &  |εu|   &  r(εu)  & |εγ|  & r(εγ)   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], εσ[nk], rεσ[nk], εu[nk], rεu[nk], εγ[nk], rεγ[nk]);
    end

    println("========================================================================")
    println("   DoF  &    h   &  |εp|   &  r(p)  &  |εz|   &  r(z)                   ")
    println("========================================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], εp[nk], rεp[nk], εz[nk], rεz[nk]);
    end


    println("========================================================================")
  end
  convergence_test(;nkmax=3,k=0,generate_output=false)
end


