# Approximate functions between linear spaces
#
# For performance reasons, these methods are typed with concrete rather than abstract types

# TODO:
#   Look for an approximate Tucker decomposition.
#   Option to use BasisFunction
#   Finish writing the tests.
#   Document the args
#   Consistent keywords for tensor deomposition tolerance

include("QOL.jl")
using ApproxFun
using AAA # Bodge
using TensorToolbox
using IterTools: product
using Flatten
using SplitApplyCombine: combinedims

struct UnivariateApproximationScheme
    sample_points::Vector{Float64}
    approximate::Function # :: Vector{Float} -> (Float -> Float)
end

function chebfun(nbr_nodes::Int64)#={{{=#
    return UnivariateApproximationScheme(
        points(Chebyshev(), nbr_nodes),
        pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev())
        )
end#=}}}=#

function aaa_wrapper(nbr_nodes::Int64)#={{{=#
    n1 = Int(round(nbr_nodes / 2))
    n2 = nbr_nodes - n1
    nodes = [collect(-1.0:n1:1.0)..., points(Chebyshev(), n2)...]

    return UnivariateApproximationScheme(
        nodes,
        first ∘ pa(aaa, nodes; pos=2)
        )
end#=}}}=#

using PyCall: (pyimport)
teneva = pyimport("teneva")

"""
    Approximate a TT decomposition of a tensor G without using the full tensor. G is hence represented as a map from m-tuples of integers between 1 and N to the reals.
"""
function TTsvd_incomplete(#={{{=#
    G::Function, # :: (1:n_1) x ... x (1:n_m) -> R
    valence::Vector{Int64};
    reqrank::Int64=10, # TODO: If I rewrite this, make reqrank a tuple of Ints
    kwargs...
    )::TTtensor

    Is, idx, idx_many = teneva.sample_tt(valence, r=reqrank)
    Gs = [G(I .+ 1) for I in eachrow(Is)]
    return TTtensor(teneva.svd_incomplete(Is, Gs, idx, idx_many, r=reqrank; kwargs...))
end#=}}}=#

function TTsvd_cross(#={{{=#
    G::Function, # :: (1:n_1) x ... x (1:n_m) -> R
    valence::Vector{Int64};
    reqrank::Int64=10,
    kwargs...
    )::TTtensor

    return TTtensor(teneva.cross( G, teneva.rand(valence, reqrank), e=1e-10, kwargs...))
end#=}}}=#

"""
    Approximate a multivariate scalar-valued function using a tensorized `univariate_approximate`.
    Available tensor decomposition methods are `hosvd` (complete), `TTsvd` (complete), `TTsvd_incomplete` (incomplete), `TTsvd_cross` (incomplete), `cp_als` (incomplete?).
"""
function approximate_scalar(#={{{=#
    m::Int64,
    g::Function; # :: [-1, 1]^m -> R
    decomposition_method=hosvd,
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    return approximate_scalar(
        m,
        g,
        decomposition_method;
        univariate_scheme=univariate_scheme,
        kwargs...
        )
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(hosvd);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    # Evaluate g on product grid
    # G_ijk = g(t_i, t_j, t_k)
    # and decompose
    # G_ijk = C^abc U1_ai U2_bj U3_ck
    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::ttensor = hosvd(G; kwargs...)
    C::Array{Float64, m} = G_decomposed.cten
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
    us::Vector{Array{Function, 2}} = Vector{Array{Function, 2}}(undef, m)
    for i in 1:m
        us[i] = mapslices(
            univariate_approximate,
            Us[i];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
   
        # Evaluate chebfuns and contract
        return only(full(ttensor(
            C,
            [map(f -> f(t), u) for (u, t) in zip(us, x)]
            )))
    end

    return g_approx
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(TTsvd);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::TTtensor = TTsvd(G; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
   
        return only(full(TTtensor(
            [map(f -> f(t), c) for (c, t) in zip(cs, x)]
            )))
    end

    return g_approx
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(TTsvd_incomplete);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    G(I::Vector{Int64})::Float64 = g([sample_points[i + 1] for i in I])
    valence = repeat([length(sample_points)], m)
    G_decomposed::TTtensor = TTsvd_incomplete(G, valence; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
    
        return only(full(TTtensor(
            [map(f -> f(t), c) for (c, t) in zip(cs, x)]
            )))
    end

    return g_approx
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(TTsvd_cross);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    G(I::Vector{Int64})::Float64 = g([sample_points[i + 1] for i in I])
    G(Is::Matrix{Int64}) = [G(Is[i, :]) for i in 1:size(Is, 1)]
    valence = repeat([length(sample_points)], m)
    G_decomposed::TTtensor = TTsvd_cross(G, valence; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
    
        # Evaluate chebfuns and contract
        return only(full(TTtensor(
            [map(f -> f(t), c) for (c, t) in zip(cs, x)]
            )))
    end

    return g_approx
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(cp_als);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    resolution = length(sample_points)
    grid = [sample_points[collect(I)] for I in product(repeat([1:resolution], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::ktensor = cp_als(G, 2 * resolution; tol=1e-10, kwargs...) # TODO: How to choose number of terms??
    lambdas::Vector{Float64} = G_decomposed.lambda
    Vs::Vector{Array{Float64, 2}} = G_decomposed.fmat

    vs::Vector{Array{Function, 2}} = Vector{Array{Function, 2}}(undef, m)
    for i in 1:m
        vs[i] = mapslices(
            univariate_approximate,
            Vs[i];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Float64
        @assert(length(x) == m)
   
        return only(full(ktensor(
            lambdas,
            [map(f -> f(t), v) for (v, t) in zip(vs, x)]
            )))
    end

    return g_approx
end#=}}}=#

"""
    Approximate a multivariate vector-valued function using a tensorized `univariate_approximate`.
    Available tensor decomposition methods are `hosvd` (complete), `TTsvd` (complete), `TTsvd_incomplete` (incomplete), `TTsvd_cross` (incomplete), `cp_als` (incomplete?).
"""
function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function; # :: [-1, 1]^m -> R^n
    decomposition_method=hosvd,
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    return approximate_vector(
        m,
        n,
        g,
        decomposition_method;
        univariate_scheme=univariate_scheme,
        kwargs...
        )
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function, # : [-1, 1]^m -> R^n
    ::typeof(hosvd);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    # Evaluate g on product grid
    # G^l_ijk = g(t_i, t_j, t_k)
    # and decompose
    # G^l_ijk = C_d^abc U1_ia U2_jb U3_kc U4^l_d
    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m + 1} = combinedims(g.(grid))
    G_decomposed::ttensor = hosvd(G; kwargs...)

    C::Array{Float64, m + 1} = G_decomposed.cten
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat

    # ghat(x, y, z) = C_d^abc u1_a(x) u2_b(y) u3_c(z) U4^l_d
    us::Vector{Array{Function, 2}} = Vector{Array{Function, 2}}(undef, m)
    for i in 1:m
        us[i] = mapslices(
            univariate_approximate,
            Us[i + 1];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
   
        
        # Evaluate chebfuns and contract
        return full(ttensor(
            C,
            [Us[1], [map(f -> f(t), u) for (u, t) in zip(us, x)]...]
            ))[:]
    end

    return g_approx
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function,
    ::typeof(TTsvd);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m + 1} = combinedims(g.(grid))
    G_decomposed::TTtensor = TTsvd(G; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i + 1];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        return full(TTtensor(
            [Cs[1], [map(f -> f(t), c) for (c, t) in zip(cs, x)]...]
            ))
    end

    return g_approx
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function,
    ::typeof(TTsvd_incomplete);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    G(I::Vector{Int64})::Float64 = g([sample_points[i + 1] for i in I[2:end]])[I[1]]
    valence = [n, repeat([length(sample_points)], m)...]
    G_decomposed::TTtensor = TTsvd_incomplete(G, valence; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i + 1];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        return full(TTtensor(
            [Cs[1], [map(f -> f(t), c) for (c, t) in zip(cs, x)]...]
            ))
    end

    return g_approx
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function,
    ::typeof(TTsvd_cross);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    G(I::Vector{Int64})::Float64 = g([sample_points[i + 1] for i in I[2:end]])[I[1] + 1]
    G(Is::Matrix{Int64}) = [G(Is[i, :]) for i in 1:size(Is, 1)]
    valence = [n, repeat([length(sample_points)], m)...]
    G_decomposed::TTtensor = TTsvd_cross(G, valence; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    cs::Vector{Array{Function, 3}} = Vector{Array{Function, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            univariate_approximate,
            Cs[i + 1];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        return full(TTtensor(
            [Cs[1], [map(f -> f(t), c) for (c, t) in zip(cs, x)]...]
            ))
    end

    return g_approx
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function,
    ::typeof(cp_als);
    univariate_scheme::UnivariateApproximationScheme=chebfun(20),
    kwargs...
    )::Function

    sample_points = univariate_scheme.sample_points
    univariate_approximate = univariate_scheme.approximate

    resolution = length(sample_points)
    grid = [sample_points[collect(I)] for I in product(repeat([1:resolution], m)...)]
    G::Array{Float64, m + 1} = combinedims(g.(grid))
    G_decomposed::ktensor = cp_als(G, 2 * resolution; tol=1e-10, kwargs...) # TODO: How to choose number of terms??
    lambdas::Vector{Float64} = G_decomposed.lambda
    Vs::Vector{Array{Float64, 2}} = G_decomposed.fmat

    vs::Vector{Array{Function, 2}} = Vector{Array{Function, 2}}(undef, m)
    for i in 1:m
        vs[i] = mapslices(
            univariate_approximate,
            Vs[i + 1];
            dims=1
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
   
        return [full(ktensor(
            lambdas,
            [Vs[1], [map(f -> f(t), v) for (v, t) in zip(vs, x)]...]
            ))...]
    end

    return g_approx
end#=}}}=#
