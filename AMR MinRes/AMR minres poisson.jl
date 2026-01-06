using Gridap, Gridap.Geometry, Gridap.Adaptivity
using DataStructures

# manufactured solution with singularity at inner corner
ϵ = 1e-2
r(x) = ((x[1]-0.5)^2 + (x[2]-0.5)^2)^(1/2)
u_exact(x) = 1.0 / (ϵ + r(x))

# generating the L-shaped domain [0,1]²\[0.5,1]x[0.5,1]
function LShapedModel(n)
  model = CartesianDiscreteModel((0,1,0,1),(n,n))
  cell_coords = map(mean,get_cell_coordinates(model))
  l_shape_filter(x) = (x[1] < 0.5) || (x[2] < 0.5)
  mask = map(l_shape_filter,cell_coords)
  model = simplexify(DiscreteModelPortion(model,mask))

  grid = get_grid(model)
  topo = get_grid_topology(model)
  return UnstructuredDiscreteModel(grid, topo, FaceLabeling(topo))
end


function amr_step(model,u_exact;order)

 orderV=2
 orderU=1

 V = FESpace(model,ReferenceFE(lagrangian,Float64,orderV),
            conformity=:H1, dirichlet_tags="boundary")
 U = FESpace(model,ReferenceFE(lagrangian,Float64,orderU),
            conformity=:H1, dirichlet_tags="boundary")

 V_tr = TrialFESpace(V,0.0)
 U_tr = TrialFESpace(U,u_exact)

 trial = MultiFieldFESpace([V_tr,U_tr])
 test = MultiFieldFESpace([V,U])

  "Setup integration measures"
  Ω = Triangulation(model)
  Γ = Boundary(model)

  dΩ = Measure(Ω,4*order)
  dΓ = Measure(Γ,2*order)

  "Get normal vectors for boundary"
  nΓ = get_normal_vector(Γ)

  "Define the weak form"
  ∇u(x)  = ∇(u_exact)(x)
  f(x)   = -Δ(u_exact)(x)

  a(u,v) = ∫(u*v + ∇(u)⋅∇(v))dΩ # full H₁ norm

  b((ε,u),(v,z))= a(ε,v) + ∫(∇(v)⋅∇(u) + ∇(ε)⋅∇(z))dΩ
  l((v,z))   = ∫(f*v)dΩ - ∫((∇u⋅nΓ)*v)dΓ     

  "Solve the FE problem"
  op = AffineFEOperator(b,l,trial,test)
  xh = solve(op)
  εh, uh = xh  
  
  "Riesz representative of the residual as error estimator η"
  h1_norm_dΩ(v) = a(v,v)
  η = estimate(h1_norm_dΩ,εh) 

  "Mark cells for refinement; those containing a fixed fraction (0.9) of the total error"
  m = DorflerMarking(0.9)
  I = Adaptivity.mark(m,η)

  "Refine the mesh using newest vertex bisection"
  method = Adaptivity.NVBRefinement(model)
  amodel = refine(method,model;cells_to_refine=I)
  fmodel = Adaptivity.get_model(amodel)

  "Compute the global error for convergence testing"
  error = sum(h1_norm_dΩ(uh - u_exact))
  return fmodel, uh, η, I, error
end

# Running the AMR by calling the function above

nsteps = 5
order = 1
model = LShapedModel(10)
mkpath("output_path")

errors = Array{Float64}(undef,nsteps)

for i in 1:nsteps
  fmodel, uh, η, I, error = amr_step(model,u_exact;order)

  # Refinement markers
  is_refined = map(i -> ifelse(i ∈ I, 1, -1), 1:num_cells(model))

  Ω = Triangulation(model)
  writevtk(
    Ω,"output_path/model_$(i-1)",append=false,
    cellfields = [
      "uh" => uh,                    # Computed solution
      "η" => CellField(η,Ω),        # Error indicators
      "is_refined" => CellField(is_refined,Ω),  # Refinement markers
      "u_exact" => CellField(u_exact,Ω),       # Exact solution
    ],
  )

  println("Error: $error, Error η: $(sum(η))")
  
  errors[i] = error
  global model = fmodel
end