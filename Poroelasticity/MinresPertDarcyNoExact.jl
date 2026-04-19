# minres perturbed Darcy; attempt with no exact solution
module PerturbedDarcyMinRes
  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra
  using DataFrames, CSV
  using Gridap.CellData

  push!(LOAD_PATH, joinpath(@__DIR__, "src"))
  using GridapMixedViscoelasticityReactionDiffusion

    K_component = 1#e-9
    K = TensorValue(K_component, 0.0, 0.0, K_component) # this will be variable later...
    Kinv = TensorValue(1/K_component, 0.0, 0.0, 1/K_component) # and so will this.

    const s_0 = 1#e-3

    print("s_0 = $(s_0)\n")
    print("K   = $(K_component)\n")

  fex(x) = VectorValue(1.0,1.0)
  gex(x) = 1.0
  pex(x)  = 0.0    # this is how we deal with no exact p? 

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
  Zh = TrialFESpace(Sh_) # when using no exact solutions, how do we define the essential BC here?

  rPh = TrialFESpace(Gh_1)
  rZh = TrialFESpace(Sh_1)

  Y = MultiFieldFESpace([Gh_1,Sh_1,Gh_,Sh_])
  X = MultiFieldFESpace([rPh,rZh,Ph,Zh])

  a3(p,q) = ∫(s_0*(p*q))dΩ
 
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 

  c(z,w) = ∫((Kinv⋅z)⋅w)dΩ 

  d(z,w) = ∫(z⋅w + (∇⋅z)*(∇⋅w))dΩ # Hdiv inner product
  d1(z,w) = c(z,w) + ∫((1\K_component)*(∇⋅z)*(∇⋅w))dΩ 
  d2(z,w) = c(z,w) + ∫((∇⋅z)*(∇⋅w))dΩ 

  e(p,q) = ∫(p*q)dΩ
  e1(p,q) = ∫((K_component)*p*q)dΩ
  e2(p,q) = ∫((K_component+s_0)*p*q)dΩ
  e3(p,q) = ∫((K_component)*(1+s_0)*p*q)dΩ
  e4(p,q) =  ∫((1+K_component*s_0)*p*q)dΩ
  e5(p,q) = ∫((1+s_0)*p*q)dΩ

  F(q) = ∫(gex*q)dΩ 
  G(w) = ∫(pex*(w⋅n_ΓD))dΓD - ∫(fex⋅w)dΩ
  
  lhs((εp,εz,p,z),(q,w,φp,φz)) =  a3(p,q) + b2(p,w) + b2(q,z) - c(z,w) + a3(φp,εp) + b2(φp,εz) + b2(εp,φz) - c(φz,εz) + e2(εp,q) + d1(εz,w) 
  rhs((q,w,φp,φz)) =  F(q) + G(w)
  
  op = AffineFEOperator(lhs,rhs,X,Y)
  εph, εzh, ph, zh = solve(op)

  # mkpath("poroelasticity_output")
  # if generate_output
  #     writevtk(Ω,"poroelasticity_output/convergence_AFW=$(num_cells(model))",order=1,
  #           cellfields=["σ1"=>σh1,"σ2"=>σh2, "p"=>ph,  "u"=>uh, "γ"=>γh, "z"=>zh])
  #     writevtk(model,"poroelasticity_output/model")
  # end
   εph, εzh, ph, zh, rPh, rZh, Ph, Zh, dΩ, dΓD, Gridap.FESpaces.num_free_dofs(X)
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

    modelfine = generate_model_unit_square(nkmax+1)
    setup_model_labels_unit_square!(modelfine)
    εpref,εzref,pref,zref,_,_,_,_,_,_,_= solve_darcy(modelfine; k=k, generate_output=false)

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_unit_square!(model)
      
       εph, εzh, ph, zh, rPh, rZh, Ph, Zh, dΩ, dΓD, ndofs=solve_darcy(model; k=k, generate_output=generate_output)

       pr_ = Interpolable(pref)
       zr_ = Interpolable(zref)
       εpr_ = Interpolable(εpref)
       εzr_ = Interpolable(εzref)
       pr = interpolate_everywhere(pr_, Ph)
       zr = interpolate_everywhere(zr_, Zh)
       εpr = interpolate_everywhere(εpr_, rPh)
       εzr = interpolate_everywhere(εzr_, rZh)

       function norm_p(p,dΩ)
         errorp = sqrt(sum(∫((K_component+s_0)*(p*p))dΩ))
         errorp
       end

       function norm_z(z,dΩ)
         errorz = sqrt(sum(∫((Kinv⋅z)⋅z)dΩ + ∫((1\K_component)*(∇⋅z)*(∇⋅z))dΩ))
         errorz
       end
        
         error_p = norm_p(pr-ph,dΩ)
         error_z = norm_z(zr-zh,dΩ)
         size_εph = norm_p(εpr-εph,dΩ)
         size_εzh = norm_z(εzr-εzh,dΩ)
       
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

    df1 = DataFrame(DOFs = nn, error_z = ez,rate_z = rz, error_p = ep,rate_p = rp, error_epsz = εz, rate_epsz = rεz, error_epsp = εp, rate_epsp = rεp)
    CSV.write("newform_minres_pert_darcy.dat", df1)
  end
  convergence_test(;nkmax=5,k=0,generate_output=false)
end