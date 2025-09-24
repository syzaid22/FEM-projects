

using Gridap
using Plots
using CategoricalArrays


function fe_solver(u,N,p)

Ω = (0.0, 1.0)
n = (N,)             # defines our number of cells
model = CartesianDiscreteModel(Ω, n)
tri = get_triangulation(model)

degree = 8*(p-1)     # degree of exactness of our quadrature
dΩ = Measure(tri,degree)

f(x) = -Δ(u)(x)

reffe = ReferenceFE(lagrangian,Float64,p)

Vₕ = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")

Uₕ = TrialFESpace(Vₕ,u)

a(u,v) = ∫(∇(u)⋅∇(v))dΩ
l(v) = ∫(v*f)dΩ

# set up linear system to be solved in form F(x)=Ax-b
Fₕ = AffineFEOperator(a,l,Uₕ,Vₕ)
uₕ = solve(Fₕ)

e = u - uₕ

el2 = sqrt( sum( ∫( e*e )dΩ ) )

eh1 = sqrt( sum(∫( e*e + ∇(e)⋅∇(e))dΩ ) )

return el2, eh1

end

function array_fe_solver(u,N_array,p)

    Nsize = size(N_array)

    el2_array = zeros(Nsize[1])
    eh1_array = zeros(Nsize[1])

    for i = 1:Nsize[1]

        errors = fe_solver(u,N_array[i],p)

        el2_array[i] = errors[1]
        eh1_array[i] = errors[2]

    end

    return el2_array, eh1_array

end
        


lm(@formula(y ~ x), DataFrame(y = log.(N_array), x = log.(el2_array)))

