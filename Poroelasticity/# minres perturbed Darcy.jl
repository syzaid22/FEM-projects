# minres perturbed Darcy
module PerturbedDarcyMinRes
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra

  push!(LOAD_PATH, joinpath(@__DIR__, "src"))
  using GridapMixedViscoelasticityReactionDiffusion

    K_component = 1e-3
    K = TensorValue(K_component, 0.0, 0.0, K_component) # this will be variable later...
    Kinv = TensorValue(1/K_component, 0.0, 0.0, 1/K_component) # and so will this.

    const s_0 = 1

    print("s_0 = $(s_0)\n")
    print("K   = $(K_component)\n")

  pex(x) = 0.1*cos(π*x[1])*sin(π*x[2])+0.15*x[1]^2
  zex(x) = -(K⋅∇(pex)(x))

  gex(x) = s_0*pex(x) + (∇⋅zex)(x) 

  function solve_darcy(model; k = k, generate_output=false)

  # Reference FEs
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

  reffe_σ_1 = ReferenceFE(bdm,Float64,k+2)
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

  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)

  Sh_1 = TestFESpace(model,reffe_σ_1,dirichlet_tags="Gamma_sig",conformity=:HDiv)
  Gh_1 = TestFESpace(model,reffe_γ_1,conformity=:L2)

  Ph = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,zex)

  rPh = TrialFESpace(Gh_1)
  rZh = TrialFESpace(Sh_1)

  Y = MultiFieldFESpace([Gh_1,Sh_1,Gh_,Sh_])
  X = MultiFieldFESpace([rPh,rZh,Ph,Zh])

  a3(p,q) = ∫(s_0*(p*q))dΩ
 
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 

  c(z,w) = ∫((Kinv⋅z)⋅w)dΩ 

  d(z,w) = ∫(z⋅w + (∇⋅z)*(∇⋅w))dΩ # Hdiv inner product
  d2(z,w) = c(z,w) + ∫((∇⋅z)*(∇⋅w))dΩ # scaled Hdiv inner product


  F(q) = ∫(gex*q)dΩ 
  G(w) = ∫(pex*(w⋅n_ΓD))dΓD

  lhs((εp,εz,p,z),(q,w,φp,φz)) =  a3(p,q) + b2(p,w) + b2(q,z) - c(z,w) + a3(φp,εp) + b2(φp,εz) + b2(εp,φz) - c(φz,εz) + ∫(K_component*εp*q)dΩ + d2(εz,w) 
  rhs((q,w,φp,φz)) =  F(q) + G(w)

  op = AffineFEOperator(lhs,rhs,X,Y)
  εph, εzh, ph, zh = solve(op)

  # mkpath("poroelasticity_output")
  # if generate_output
  #     writevtk(Ω,"poroelasticity_output/convergence_AFW=$(num_cells(model))",order=1,
  #           cellfields=["σ1"=>σh1,"σ2"=>σh2, "p"=>ph,  "u"=>uh, "γ"=>γh, "z"=>zh])
  #     writevtk(model,"poroelasticity_output/model")
  # end

  eph = pex-ph
  ezh = zex-zh

  error_p = sqrt(sum(∫(K_component*eph*eph)dΩ))
  error_z = sqrt(sum(d2(ezh,ezh)))

  size_εph = sqrt(sum(∫(K_component*εph*εph)dΩ))
  size_εzh = sqrt(sum(d2(εzh,εzh)))

  size_εph,size_εzh,error_p,error_z, Gridap.FESpaces.num_free_dofs(X)
  end



  function  convergence_test(; nkmax, k=0, generate_output=false)
    ep   = Float64[]
    rp   = Float64[] 
    ez   = Float64[]
    rz   = Float64[]

    εp   = Float64[]
    rεp   = Float64[] 
    εz   = Float64[]
    rεz   = Float64[]

    push!(rp,0.)
    push!(rz,0.)

    push!(rεp,0.)
    push!(rεz,0.)

    nn   = Int[]
    hh   = Float64[]

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_unit_square!(model)
      
       size_εph,size_εzh,error_p,error_z, ndofs=solve_darcy(model; k=k, generate_output=generate_output)
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk) # i.e. using diameter of simplices

       push!(ep,error_p)
       push!(ez,error_z)

       push!(εp,size_εph)
       push!(εz,size_εzh)


       if nk>1
         push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rz, log(ez[nk]/ez[nk-1])/log(hh[nk]/hh[nk-1]))

         push!(rεp, log(εp[nk]/εp[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rεz, log(εz[nk]/εz[nk-1])/log(hh[nk]/hh[nk-1]))
       end
    end

    println("=========================================================")
    println("   DoF  &    h   &  e(p)   &  r(p)  &  e(z)   &  r(z)    ")
    println("=========================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], ep[nk], rp[nk], ez[nk], rz[nk]);
    end

     println("=========================================================")
    println("   DoF  &    h   &   εp   &  r(εp)  &   εz   &  r(εz)    ")
    println("=========================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], εp[nk], rεp[nk], εz[nk], rεz[nk]);
    end

    println("==========================================================")
  end
  convergence_test(;nkmax=6,k=0,generate_output=false)
end