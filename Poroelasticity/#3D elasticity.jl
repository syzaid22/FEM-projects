#3D elasticity
# code is only pseudo-working; γ doesn't converge
module ElasticityMixedTensor_mixedBCTests
  using Gridap
  #using GridapMixedViscoelasticityReactionDiffusion
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  push!(LOAD_PATH, joinpath(@__DIR__, "src"))
  using GridapMixedViscoelasticityReactionDiffusion

  # Material parameters
  # const E = 1.0e2
  # const ν = 0.49
  # const λ = (E*ν)/((1+ν)*(1-2*ν))
  # const μ = E/(2*(1+ν))
  const λ = 1
  const μ = 1

  const d = 3

    print("λ  = $(λ)\n")
    print("μ  = $(μ)\n")
 
  calC(τ) = 2*μ*τ + λ*tr(τ)*one(τ)

  uex(x) = VectorValue(-cos(x[2])*x[1],x[2]*sin(x[1]),-(x[2]^2)*sin(x[1]))
  σex(x) = (calC∘ε(uex))(x)
  γex(x) = 0.5*(∇(uex)(x) - transpose(∇(uex)(x)))
  fex(x) = -(∇⋅σex)(x)

  comp1=extract_component(1)
  comp2=extract_component(2)
  comp3=extract_component(3)
  row1=extract_row3d(1)
  row2=extract_row3d(2)
  row3=extract_row3d(3)

  function solve_elasticityMixed(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},k)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 5+k
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  # Boundary triangulations and outer unit normals
  #Γσ = BoundaryTriangulation(model,tags = "Gamma_sig")
  Γu = BoundaryTriangulation(model,tags = "Gamma_u")
  #n_Γσ = get_normal_vector(Γσ) 
  n_Γu = get_normal_vector(Γu)
  #dΓσ = Measure(Γσ,degree)
  dΓu = Measure(Γu,degree)

  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Vh_ = TestFESpace(model,reffe_u,conformity=:L2)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)

  Sh1 = TrialFESpace(Sh_,row1∘σex) 
  Sh2 = TrialFESpace(Sh_,row2∘σex)
  Sh3 = TrialFESpace(Sh_,row3∘σex)
  Vh = TrialFESpace(Vh_)
  Gh1 = TrialFESpace(Gh_)
  Gh2 = TrialFESpace(Gh_)
  Gh3 = TrialFESpace(Gh_)

  Y = MultiFieldFESpace([Sh_,Sh_,Sh_,Vh_,Gh_,Gh_,Gh_])
  X = MultiFieldFESpace([Sh1,Sh2,Sh3,Vh,Gh1,Gh2,Gh3])

  a((σ1,σ2,σ3),(τ1,τ2,τ3)) = ∫(1/(2*μ)*(σ1⋅τ1 + σ2⋅τ2 + σ3⋅τ3))dΩ -
                   ∫(λ/(2*μ*(2*μ+ d*λ))*(comp1∘σ1+comp2∘σ2+comp3∘σ3)*(comp1∘τ1+comp2∘τ2+comp3∘τ3))dΩ
 
  b((τ1,τ2,τ3),(v,η1,η2,η3)) = ∫((comp1∘v)*(∇⋅τ1)+(comp2∘v)*(∇⋅τ2)+(comp3∘v)*(∇⋅τ3))dΩ + ∫(η1*(comp2∘τ1-comp1∘τ2) + η2*(comp3∘τ1-comp1∘τ3) + η3*(comp2∘τ3-comp3∘τ2))dΩ 

  F(τ1,τ2,τ3) =  ∫((τ1⋅n_Γu)*(comp1∘uex) + (τ2⋅n_Γu)*(comp2∘uex) + (τ3⋅n_Γu)*(comp3∘uex))dΓu 
  G(v) = ∫(fex⋅v)dΩ

  lhs((σ1,σ2,σ3,u,γ1,γ2,γ3),(τ1,τ2,τ3,v,η1,η2,η3)) =  a((σ1,σ2,σ3),(τ1,τ2,τ3)) + b((τ1,τ2,τ3),(u,γ1,γ2,γ3)) + b((σ1,σ2,σ3),(v,η1,η2,η3))
  rhs((τ1,τ2,τ3,v,η1,η2,η3)) =  F(τ1,τ2,τ3) - G(v) 

  op = AffineFEOperator(lhs,rhs,X,Y) 
  σh1, σh2, σh3, uh, γh1, γh2, γh3 = solve(op)

  # if generate_output
  #   vtk_dir = joinpath(@__DIR__, "paraview-data")
  #     writevtk(Ω,joinpath(vtk_dir,"convergence_AFW=$(num_cells(model))"),order=1,
  #           cellfields=["σ1"=>σh1,"σ2"=>σh2, "u"=>uh, "γ"=>γh])
  #     writevtk(model,joinpath(vtk_dir,"model"))
  # end

  eσ1h = (row1∘σex)-σh1
  eσ2h = (row2∘σex)-σh2
  eσ3h = (row3∘σex)-σh3
  euh  = uex-uh
  eγ1h  = comp2∘row1∘γex-γh1
  eγ2h  = comp1∘row3∘γex-γh2
  eγ3h  = comp3∘row2∘γex-γh3
 error_σ = sqrt(sum(∫(eσ1h⋅eσ1h+eσ2h⋅eσ2h+eσ3h⋅eσ3h)dΩ +
                     ∫((∇⋅eσ1h)*(∇⋅eσ1h)+(∇⋅eσ2h)*(∇⋅eσ2h)+(∇⋅eσ3h)*(∇⋅eσ3h))dΩ))
  error_u = sqrt(sum(∫(euh⋅euh)dΩ))
  error_γ = sqrt(sum(∫(eγ1h*eγ1h + eγ2h*eγ2h + eγ3h*eγ3h)dΩ))

  error_σ,error_u, error_γ, Gridap.FESpaces.num_free_dofs(X)
  end

  function  convergence_test(; nkmax, k=0, generate_output=false)
    eσ   = Float64[]
    rσ   = Float64[]
    eu   = Float64[]
    ru   = Float64[]
    eγ   = Float64[]
    rγ   = Float64[]
    push!(ru,0.)
    push!(rσ,0.)
    push!(rγ,0.)
    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_cube(nk) # Discrete model
       setup_model_labels_unit_cube!(model)
      
       error_σ, error_u, error_γ, ndofs=solve_elasticityMixed(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk)
       push!(eσ,error_σ)
       push!(eu,error_u)
       push!(eγ,error_γ)

       if nk>1
         push!(rσ, log(eσ[nk]/eσ[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rγ, log(eγ[nk]/eγ[nk-1])/log(hh[nk]/hh[nk-1]))
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
  end
  convergence_test(;nkmax=4,k=0,generate_output=true)
end

