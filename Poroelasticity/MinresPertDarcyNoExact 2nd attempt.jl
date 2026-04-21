# perturbed Darcy; attempt with no exact solution

  using Gridap
  import Gridap: ∇
  using Printf
  using LinearAlgebra
  #using DataFrames, CSV
  using Gridap.CellData
 
  function generate_model_unit_square(nk)
    domain =(0,1,0,1)
    n      = 2^nk
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
  end

  function setup_model_labels_mixed!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Gamma_out",[8]) #right 
    add_tag!(labels,"Gamma_in",[7,1,3]) # left and 2 corners
    add_tag!(labels,"Gamma_topbot",[1,2,3,4,5,6]) # corners bottom and top

  end  

  κ = 1.0 
  s_0 = 1.0

  load = VectorValue(1,0)
  zerov = VectorValue(0.0,0.0)
  g = 1.0
  zΓ = VectorValue(1.0,0.0)

  function solve_darcy(model; k = k, generate_output=false)

  # Reference FEs
  #reffe_σ = ReferenceFE(raviart_thomas,Float64,k)
  reffe_σ = ReferenceFE(bdm,Float64,k+1)
  reffe_γ = ReferenceFE(lagrangian,Float64,k)

  # Numerical integration
  degree = 2*(2+k)
  Ω = Interior(model)
  dΩ = Measure(Ω,degree)

  Sh_ = TestFESpace(model,reffe_σ,dirichlet_tags=["Gamma_in","Gamma_topbot"],conformity=:HDiv)
  Gh_ = TestFESpace(model,reffe_γ,conformity=:L2)

  Ph = TrialFESpace(Gh_)
  Zh = TrialFESpace(Sh_,[zΓ,zerov]) # we pass to the trial space just the prescribed BC 
 
  Y = MultiFieldFESpace([Gh_,Sh_])
  X = MultiFieldFESpace([Ph,Zh])

  a3(p,q) = ∫(s_0*(p*q))dΩ
  b2(q,w) = ∫( q*(∇⋅w) )dΩ 
  c(z,w) = ∫((1.0/κ)*(z⋅w))dΩ 
 
  F(q) = ∫(g*q)dΩ 
  G(w) = ∫(load⋅w)dΩ
  
  lhs((p,z),(q,w)) =  a3(p,q) + b2(p,w) + b2(q,z) - c(z,w) 
  rhs((q,w)) =  F(q) - G(w)
  
  op = AffineFEOperator(lhs,rhs,X,Y)
  ph, zh = solve(op)
 
  ph, zh, Ph, Zh, dΩ, Gridap.FESpaces.num_free_dofs(X)
  end

  function  convergence_test(; nkmax, k=0, generate_output=false)
    ep   = Float64[]
    rp   = Float64[] 
    ez   = Float64[]
    rz   = Float64[]
 
    push!(rp,0.)
    push!(rz,0.)
 
    nn   = Int[]
    hh   = Float64[]

    modelfine = generate_model_unit_square(nkmax+3)
    setup_model_labels_mixed!(modelfine)
    pref,zref,_,_,_,_= solve_darcy(modelfine; k=k, generate_output=false)

    for nk in 1:nkmax
       println("******** Refinement step: $nk")
       model=generate_model_unit_square(nk) # Discrete model
       setup_model_labels_mixed!(model)
      
       ph, zh, Ph, Zh, dΩ, ndofs=solve_darcy(model; k=k, generate_output=generate_output)

       pr_ = Interpolable(pref)
       zr_ = Interpolable(zref) 
       pr = interpolate_everywhere(pr_, Ph)
       zr = interpolate_everywhere(zr_, Zh) 

       function norm_p(p,dΩ)
         errorp = sqrt(sum(∫((p*p))dΩ))
         errorp
       end

       function norm_z(z,dΩ)
         errorz = sqrt(sum(∫((z⋅z))dΩ + ∫((∇⋅z)*(∇⋅z))dΩ))
         errorz
       end
        
         error_p = norm_p(pr-ph,dΩ)
         error_z = norm_z(zr-zh,dΩ) 
       
       push!(nn,ndofs)
       push!(hh,sqrt(2)/2^nk) # i.e. using diameter of simplices

       push!(ep,error_p)
       push!(ez,error_z) 


       if nk>1
         push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
         push!(rz, log(ez[nk]/ez[nk-1])/log(hh[nk]/hh[nk-1])) 
       end
    end

    println("=========================================================")
    println("   DoF  &    h   &  e(p)   &  r(p)  &  e(z)   &  r(z)    ")
    println("=========================================================")

    for nk=1:nkmax
       @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n",
                nn[nk], hh[nk], ep[nk], rp[nk], ez[nk], rz[nk]);
    end

    

    println("==========================================================")
 
  end
  convergence_test(;nkmax=5,k=0,generate_output=false)
