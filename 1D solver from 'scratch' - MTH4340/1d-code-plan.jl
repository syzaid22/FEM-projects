using FastGaussQuadrature
using Polynomials
using LinearAlgebra
using ForwardDiff # we use this to compute derivatives of the exact solution

using Plots # for plotting convergence results

# Compute the mesh for a domain (a,b), N cells, uniform 1d mesh constructor

struct Mesh
   nodes_coordinates # Returns the nodes coordinates for a given cell
   cell_nodes        # Returns the nodes global Id for a given cell
end

# constructor 
function Mesh(a,b,N) 
  h = (b-a)/N

  nodes_coordinates = [] 
  cell_nodes = []

  for i in 1:N
    coord = a+(i-1)*h # coordinate of the left node of the cell i
    node = (i,i+1) # global numbering of the nodes of the cell i
    
    push!(nodes_coordinates,coord)
    push!(cell_nodes,node)
  end

  push!(nodes_coordinates,b) # add the last node coordinate 

  return Mesh(nodes_coordinates,cell_nodes)
end

# we create methods (API) to access the mesh data
get_cell_nodes(m::Mesh)=m.cell_nodes
get_node_coordinates(m::Mesh)=m.nodes_coordinates

# Quadrature (in the reference cell)
# you can build the Quadrature using Julia package for Gauss quadrature 
# e.g., FastGaussQuadratures
struct Quadrature
  points
  weights
end

# and create a (trivial) API
get_integration_points(q::Quadrature)=q.points
get_weights(q::Quadrature)=q.weights

function Quadrature(degree)
  n = ceil(Int,(degree+1)/2) # number of points needed for exact integration; n points integrates polynomials up to degree 2n-1 exactly
  points, weights = gausslegendre(n) # use existing package
   return Quadrature(points,weights) # call default constructor 
end

# Create a 1D reference FE space in [-1,1] for an arbitrary order p
struct RefFE
  shape_functions::Vector{Polynomial}
  gradient_shape_functions::Vector{Polynomial}
end

# we use a Lagrangian basis on equidistant nodes in [-1,1]
function RefFE(p)
  sf=Vector{Polynomial}(undef,p+1) # we'll need p+1 points for order p
  gsf=Vector{Polynomial}(undef,p+1)

  nodes = Vector(range(-1, stop=1, length=p+1)) # work on [-1,1]
  
  function Lagrangian_polynomial(k) # k = 1,...,p+1; takes in index/numbering of node

    den = 1.0
    l = Polynomial(1.0)

    for i in 1:p+1
      if i != k
      den = den*(nodes[k] - nodes[i])

      l1 = Polynomial([-nodes[i],1])
      l = l*l1
      else end
    end
     l =  l/den
    return l
  end

  for j = 1:p+1 # store shape functions and their derivatives/gradients
    φ=Lagrangian_polynomial(j)
    sf[j]=φ
    gsf[j]=derivative(φ)
  end

  return RefFE(sf,gsf)
end

# Geometric mapping from the reference cell to physical cells 
struct GeoMap
  maps::Vector{Polynomial}
  jacobian::Vector{Polynomial}
end

function GeoMap(mesh::Mesh)
  cell_nodes = get_cell_nodes(mesh)
  node_coords = get_node_coordinates(mesh)
  maps = Vector{Polynomial}(undef, length(cell_nodes))
  jacobian = Vector{Polynomial}(undef, length(cell_nodes))
  
  for (icell,nodes) in enumerate(cell_nodes)
    # maps [-1,1] to [x1,x2]; i.e. K̂ to K
    x1, x2 = node_coords[nodes[1]], node_coords[nodes[2]]  
    maps[icell] = x1 * Polynomial([0.5, -0.5]) + x2 * Polynomial([0.5, 0.5])
    jacobian[icell] = derivative(maps[icell])
  end

  return GeoMap(maps,jacobian)
end


get_cell_jacobian(gm::GeoMap) = gm.jacobian
get_cell_maps(gm::GeoMap) = gm.maps


# FE Space 

# This struct starts getting a little bit complicated. As we know,
# the global FE space requires the mesh, the reference FE, a 
# local-to-global index map for assembly, and it also needs to know
# whether a node is fixed or free.

# In this work, we assume that the whole boundary, i.e., x = a and b,
# is of Dirichlet type. In order to distinguish between free and fixed
# dofs, we can use the following. We enumerate fixed nodes with 
# -1, -2, ... and free dofs with 1, 2, ... Store this local-to-global
# map (for both free and fixed dofs), in a vector. At each cell, you can
# extract the nodes (from the mesh), and access that vector in the 
# corresponding positions to get the local-global index at the cell
# level. You can alternatively just create the cell-wise local-global
# vector of vectors.
# For the fixed nodes, you have to store the values to be fixed.

struct FESpace 
  mesh::Mesh
  reffe::RefFE
  node_map::Array # local to global map
  fixed_values # e.g, a vector with values at the Dirichlet nodes 
  ndof::Int
end

function FESpace(mesh::Mesh,reffe::RefFE,uD::Function) 
   # uD is a function such that uD(a) = ua, uD(b) = ub; i.e. gives Dirichlet data
   node_coords = get_node_coordinates(mesh)
   fixed_values = [uD(node_coords[1]), uD(node_coords[end])] 

ncell = length(get_cell_nodes(mesh))
ncelldof = length(reffe.shape_functions)
node_map = Vector{Vector{Int}}(undef,ncell) # gives us the local to global map

for i in 1:ncell
  if i == 1
    node_map[i] = [-1, 1:ncelldof-1...] # takes 1:ncelldof-1 and distributes them across the vector
  else
    prev_num = node_map[i-1][end]
    node_map[i] = [prev_num:prev_num+ncelldof-1...]

  
    i == ncell && (node_map[i][end] = -2) # this labels the last node of the last cell with global number -2; uses a compact if statement
  end
end

ndof = node_map[end][end-1] # i.e. number of free dofs
return FESpace(mesh, reffe, node_map, fixed_values, ndof)

end # FESpace function

# assembly of the global matrix and vector by adding local contributions
# optional parameter τ as on/off for streamline diffusion; by default, τ=0.0 is off, otherwise on
function assemble_matrix_and_vector(f::Function,μ::Function,β::Function,fespace::FESpace,quad::Quadrature; τ=0.0) 
  reffe, ndof = fespace.reffe, fespace.ndof
  geom = GeoMap(fespace.mesh)
  cell_maps = get_cell_maps(geom)
  cell_jacs = get_cell_jacobian(geom)
  int_pts, int_wts = get_integration_points(quad), get_weights(quad)
  ncelldof = length(reffe.shape_functions)

  Aloc = zeros(ncelldof,ncelldof)
  bloc = zeros(ncelldof)

  A = zeros(ndof,ndof)
  b = zeros(ndof)

  for (icell,l2g_dof) in enumerate(fespace.node_map)

    fill!(Aloc,0.0)
    fill!(bloc,0.0)

    jac_vals = (cell_jacs[icell]).(int_pts) # evaluate the Jacobian at the integration points
    xphys = (cell_maps[icell]).(int_pts) # map the integration points to the physical cell
    μvals = μ.(xphys) # evaluate μ at the physical integration points
    βvals = β.(xphys) # evaluate β at the physical integration points

    for (i, dφi) in enumerate(reffe.gradient_shape_functions) # local matrix evaluation
      φi = reffe.shape_functions[i]

      for (j, dφj) in enumerate(reffe.gradient_shape_functions) 
        ext_μvals = iszero(τ) ? μvals : ( τ .* (βvals.^2) .+ μvals ) # modifies diffusion coeffs if τ > 0; otherwise retains μvals
        diff_evals = ext_μvals .* (dφi*dφj).(int_pts) ./ jac_vals 

        conv_evals = (φi*dφj).(int_pts) .* βvals # convection term

        Aloc[i,j] = dot(int_wts, diff_evals) + dot(int_wts, conv_evals) # add the two contributions
      end

      fvals = f.(xphys) .* φi.(int_pts) .* jac_vals
      bloc[i] = dot( int_wts, fvals ) # forcing term
    end # loop for local matrix and vector assembly

    # Assemble the local contributions into the global matrix and vector
    for (ilocal, iglobal) in enumerate(l2g_dof)
      iglobal < 0 && continue # skip fixed Dirichlet nodes (negative global indices) during assembly; we don't test against the shape functions associated with fixed dofs

        for (jlocal, jglobal) in enumerate(l2g_dof)
          if jglobal > 0 # free dof
            A[iglobal,jglobal] += Aloc[ilocal,jlocal]
          else # fixed dof
            b[iglobal] -= Aloc[ilocal,jlocal] * fespace.fixed_values[-jglobal] # from the offset function defined using Dirichlet data
          end
        end

      b[iglobal] += bloc[ilocal]
    end # for loop over local to global mapping

  end
  return A, b
end # function


function compute_L2_H1_error(mesh::Mesh,reffe::RefFE,quad::Quadrature,u_exact::Function,uh::Vector{Float64}) # we take in the full solution vector uh (including fixed dofs)

  geom = GeoMap(mesh)
  cell_maps, cell_jacs = get_cell_maps(geom), get_cell_jacobian(geom)
  int_pts, int_wts = get_integration_points(quad), get_weights(quad)

  L2sq, H1semisq = 0.0, 0.0

  N = length(get_cell_nodes(mesh))
  ncelldof = length(reffe.shape_functions)
  p = ncelldof - 1

  for icell in 1:N
    gnode_idx = (icell-1)*p .+ (1:ncelldof) # new global node indexing for full uh vector
    uh_local = uh[gnode_idx] # local solution vector

    xphys = (cell_maps[icell]).(int_pts) # map the integration points to the physical cell
    jac_vals = (cell_jacs[icell]).(int_pts) # evaluate the Jacobian at the integration points

    # these are vectors of vectors: e.g. each element of φvals contains
    # the values of one shape function evaluated at all integration points
    φvals = [(reffe.shape_functions[i]).(int_pts) for i in 1:ncelldof]
    dφvals = [(reffe.gradient_shape_functions[i]).(int_pts) for i in 1:ncelldof]

    Nq = length(int_pts)
    uh_q, ∇uh_q = zeros(Nq), zeros(Nq)

    for i in 1:ncelldof # sums up the contributions from each shape function
      uh_q .+= uh_local[i] .* φvals[i] # contributions added to each integration point separately
      ∇uh_q .+= uh_local[i] .* (dφvals[i] ./ jac_vals)
    end

    ∇u(x) = ForwardDiff.derivative(u_exact, x)
    u_q, ∇u_q = u_exact.(xphys), ∇u.(xphys)

    L2sq += dot(int_wts, (u_q - uh_q).^2 .* jac_vals)
    H1semisq += dot(int_wts, (∇u_q - ∇uh_q).^2 .* jac_vals)
    # we do this for each cell and add it to the total error
  end

  return sqrt(L2sq), sqrt(L2sq + H1semisq)
end

# function that carries out h-refinement
# it calls the previous functions to assemble and solve the system 
# for each number of cells given in N_array
# then computes the errors and returns them
# we also return the corresponding size of the meshes, the h values
function h_refinement_test(p::Int,N_array,u_exact::Function,f::Function,
  μ::Function,β::Function,domain_min::Float64,domain_max::Float64; τ=0.0)

  size = length(N_array)

  errL2_h = Array{Float64}(undef,size)
  errH1_h = Array{Float64}(undef,size)
  h_array = Array{Float64}(undef,size)

  for (i,N) in enumerate(N_array)

   mesh = Mesh(domain_min,domain_max,N)
   order = p
   reffe = RefFE(order)

   fespace = FESpace(mesh,reffe,u_exact)
   quad = Quadrature(2*order)

   h = (domain_max-domain_min)/N
   

   if τ != 0.0
      # compute ||β||_∞ by sampling β at the quadrature points on every cell
     geom = GeoMap(mesh)
     cell_maps = get_cell_maps(geom)
     int_pts = get_integration_points(quad)

     beta_max = 0.0
      for cmap in cell_maps
        xphys = cmap.(int_pts)
        beta_max = max(beta_max, maximum(abs.(β.(xphys))))
       end

     beta_inf = beta_max
     τ = beta_inf == 0.0 ? 0.0 : h / beta_inf # guard against zero to avoid division by zero
     println("Streamline diffusion on with τ = $τ")
   end

   A, b = assemble_matrix_and_vector(f,μ,β,fespace,quad; τ=τ)

   uh_int = A\b

   uh_full = zeros(length(uh_int)+2)
   uh_full[1] = fespace.fixed_values[1]
   uh_full[end] = fespace.fixed_values[2]
   uh_full[2:end-1] .= uh_int

   errL2,errH1 = compute_L2_H1_error(mesh,reffe,quad,u_exact,uh_full)

   errL2_h[i] = errL2
   errH1_h[i] = errH1
   h_array[i] = h

  end # for loop

 return errL2_h, errH1_h, h_array
end # function

# plotting convergence results against h; errors1 for L2, errors2 for H1
function convergence_plot(h_arr,errors1,errors2)
  plot(h_arr,errors1,label=["L2"],shape=:auto)
  plot!(xaxis=:log, yaxis=:log,
  shape=:auto,
  label=["errors"],
  xlabel="h",ylabel="error norm")
  plot!(h_arr,errors2,label=["H1"],shape=:auto)
end


#We can use this function to plot the solutions, but it's not asked for in the assignment

#= 
function evaluate_fe_function(mesh::Mesh,reffe::RefFE, fespace::FESpace, uh_int::Vector{Float64}; nplot=4)

  cell_maps = get_cell_maps(GeoMap(mesh)) # maps from K̂ to K
  xvals, uvals = Float64[], Float64[]

  for (icell,l2g_dof) in enumerate(fespace.node_map)

    xplot = range(-1, stop=1, length=nplot) # splits up the reference cell [-1,1]
    xcell = cell_maps[icell].(xplot) # map to physical cell

    ucell = zeros(length(xplot)) 

    # looking at one cell's local to global mapping, e.g. [ -1, 1, 2 ] for first cell in quadratic FE case
    for (ilocal, iglobal) in enumerate(l2g_dof) 
      "Should this actually be xcell instead of xplot? Probably not"
      φvals = reffe.shape_functions[ilocal].(xplot) # evaluate the i-th local shape function at evaluation points in [-1,1]
      if iglobal > 0
        # scale the i-th shape function value by the coefficient from the solution vector uh; 
        # then add the contributions from this basis function to each evaluation point separately
        ucell .+= uh_int[iglobal] .* φvals 
      else
        ucell .+= fespace.fixed_values[-iglobal] .* φvals
      end

    end

    sidx = icell == 1 ? 1 : 2 # avoid double counting of nodes
    append!(xvals,xcell[sidx:end])
    append!(uvals,ucell[sidx:end])
  end

  return xvals, uvals
end

# plotting actual solutions
u_exact(x) = sin(pi*x) 
μ(x) = 1.0 + x
β(x) = 0.0 # no convection
τ = 0.0 # streamline diffusion off
f(x) = (pi^2)*(1+x)*sin(pi*x) - pi*cos(pi*x)
domain_min = 0.0
domain_max = 1.0

mesh = Mesh(0.0, 1.0, 2^5)
order = 1
reffe = RefFE(order) # linear elements
fespace = FESpace(mesh, reffe, u_exact)
quad = Quadrature(2 * order)

A, b = assemble_matrix_and_vector(f, μ, β, fespace, quad; τ=τ)

uh_int = A \ b
xvals, uvals = evaluate_fe_function(mesh, reffe, fespace, uh_int)

plot(xvals, uvals, label="FE solution", legend=:bottomright)
plot!(xvals, u_exact.(xvals), label="Exact solution")
=#