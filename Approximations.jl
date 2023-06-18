# Approximate functions between linear spaces
#
# For performance reasons, these methods are typed with concrete rather than abstract types

# TODO:
#   Look for an approximate Tucker decomposition as well.
#   Option to use AAA instead of Chebyshev interpolation
#   Finish writing the tests.
#   Document the args

include("QOL.jl")
using ApproxFun
using AAA # Bodge
using TensorToolbox
using IterTools: (product)
using SplitApplyCombine: (combinedims)

Chebfun = Fun{Chebyshev{ChebyshevInterval{Float64}, Float64}, Float64, Vector{Float64}}

using PyCall: (pyimport)
teneva = pyimport("teneva")

"""
    function TTsvd_incomplete(
        G::Function, # :: (1:n_1) x ... x (1:n_m) -> R
        valence::Vector{Int64};
        reqrank::Int64=10,
        kwargs...
        )::TTtensor

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
    function approximate_scalar(
        m::Int64,
        g::Function; # :: [-1, 1]^m -> R
        decomposition_method=TTsvd,
        univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
        kwargs...
        )::Function

Approximate a multivariate scalar-valued function using a tensorized univariate_approximate.
Available tensor decomposition methods are `hosvd` (Tucker decomposition), `TTsvd`, `TTsvd_incomplete`, `TTsvd_cross`, `cp_als`.

"""
function approximate_scalar(#={{{=#
    m::Int64,
    g::Function; # :: [-1, 1]^m -> R
    decomposition_method=TTsvd,
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    return approximate_scalar(
        m,
        g,
        decomposition_method;
        univariate_approximate=univariate_approximate,
        kwargs...
        )
end#=}}}=#

function approximate_scalar(#={{{=#
    m::Int64,
    g::Function,
    ::typeof(hosvd);
    sample_points=points(Chebyshev(), 20),
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C^abc U1_ai U2_bj U3_ck
    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::ttensor = hosvd(G; kwargs...)

    C::Array{Float64, m} = G_decomposed.cten
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
    us::Vector{Array{Chebfun, 2}} = Vector{Array{Chebfun, 2}}(undef, m)
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
    ::typeof(TTsvd); # TODO: Was there a nicer syntax for this? Like Type{TTsvd}?
    sample_points=points(Chebyshev(), 20),
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C1^a_ib C2^b_jc C3^c_ka
    grid = [sample_points[collect(I)] for I in product(repeat([1:length(sample_points)], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::TTtensor = TTsvd(G; kwargs...)

    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
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
    ::typeof(TTsvd_incomplete);
    sample_points=points(Chebyshev(), 20),
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C1^a_ib C2^b_jc C3^c_ka
    G(I::Vector{Int64})::Float64 = g([sample_points[i] for i in I])
    valence = repeat([length(sample_points)], m)
    G_decomposed::TTtensor = TTsvd_incomplete(G, valence; kwargs...)

    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
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
    ::typeof(TTsvd_cross);
    sample_points=points(Chebyshev(), 20),
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C1^a_ib C2^b_jc C3^c_ka
    G(I::Vector{Int64})::Float64 = g([sample_points[i] for i in I])
    G(Is::Matrix{Int64}) = [G(Is[i, :]) for i in 1:size(Is, 1)]
    valence = repeat([length(sample_points)], m)
    G_decomposed::TTtensor = TTsvd_cross(G, valence; kwargs...)

    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
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
    sample_points=points(Chebyshev(), 20),
    univariate_approximate=pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # :: R^N -> (R -> R)
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C^abc U1_ai U2_bj U3_ck
    resolution = length(sample_points)
    grid = [sample_points[collect(I)] for I in product(repeat([1:resolution], m)...)]
    G::Array{Float64, m} = g.(grid)
    G_decomposed::ktensor = cp_als(G, 2 * resolution; tol=1e-10, kwargs...) # TODO: How to choose number of terms??

    lambdas::Vector{Float64} = G_decomposed.lambda
    Vs::Vector{Array{Float64, 2}} = G_decomposed.fmat

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
    vs::Vector{Array{Chebfun, 2}} = Vector{Array{Chebfun, 2}}(undef, m)
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
   
        # Evaluate chebfuns and contract
        return only(full(ktensor(
            lambdas,
            [map(f -> f(t), v) for (v, t) in zip(vs, x)]
            )))
    end

    return g_approx
end#=}}}=#

# TODO: All of these need to be updated
"""
    function approximate_vector(
        m::Int64,
        n::Int64,
        g::Function; # :: [-1, 1]^m -> R^n
        decomposition_method=TTsvd,
        res::Int64=20, # nbr of interpolation points in each direction
        kwargs...
        )::Function

Approximate a vector-valued function using approximate TT decomposition and Chebyshev interpolation
Available tensor decomposition methods are `TTsvd`, `TTsvd_incomplete`, and `hosvd` (Tucker decomposition).
"""
function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function; # :: [-1, 1]^m -> R^n
    decomposition_method=TTsvd,
    res::Int64=20, # nbr of interpolation points in each direction
    kwargs...
    )::Function

    return approximate_vector(m, n, g, decomposition_method; res=res, kwargs...)
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function, # :: [-1, 1]^m -> R^n
    ::typeof(TTsvd);
    res::Int64=20, # nbr of interpolation points in each direction
    kwargs...
    )::Function

    @assert(length(g(zeros(m))) == n)
    valence = [n, repeat([res], m)...]

    # Evaluate g on Chebyshev grid
    # G^l_ijk = g^l(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G^l_ijk = C1^al_b C2^b_ic C3^c_jd C4^d_ka
    chebpts::Vector{Float64} = chebyshevpoints(res)
    chebgrid = [chebpts[collect(I)] for I in product(repeat([1:res], m)...)]
    G::Array{Float64, m + 1} = combinedims(g.(chebgrid))
    G_decomposed::TTtensor = TTsvd(G; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat^l(x, y, z) = C1^al_b c2^b_c(x) c3^c_d(y) c4^d_a(z)
    cs::Vector{Array{Chebfun, 3}} = Vector{Array{Chebfun, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Cs[i + 1];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
        return full(TTtensor(
            [Cs[1], [map(f -> f(t), c) for (c, t) in zip(cs, x)]...]
            ))
    end

    return g_approx
end#=}}}=#

function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function, # :: [-1, 1]^m -> R^n
    ::typeof(TTsvd_incomplete);
    res::Int64=20, # nbr of interpolation points in each direction
    kwargs...
    )::Function

    @assert(length(g(zeros(m))) == n)
    valence = [n, repeat([res], m)...]

    # Evaluate g on Chebyshev grid
    # G^l_ijk = g^l(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G^l_ijk = C1^al_b C2^b_ic C3^c_jd C4^d_ka
    chebpts::Vector{Float64} = chebyshevpoints(res)
    G(I::Vector{Int64})::Float64 = g([chebpts[i] for i in I[2:end]])[I[1]]
    G_decomposed::TTtensor = TTsvd_incomplete(G, valence; kwargs...)
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat^l(x, y, z) = C1^al_b c2^b_c(x) c3^c_d(y) c4^d_a(z)
    cs::Vector{Array{Chebfun, 3}} = Vector{Array{Chebfun, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
            Cs[i + 1];
            dims=2
            )
    end

    function g_approx(
        x::Vector{Float64}
        )::Vector{Float64}
        @assert(length(x) == m)
        
        # Evaluate chebfuns and contract
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
    ::typeof(hosvd); # TODO: Was there a nicer syntax for this? Like Type{TTsvd}?
    res::Int64=20,
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G^l_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G^l_ijk = C_d^abc U1_ia U2_jb U3_kc U4^l_d
    chebpts::Vector{Float64} = chebyshevpoints(res)
    chebgrid = [chebpts[collect(I)] for I in product(repeat([1:res], m)...)]
    G::Array{Float64, m + 1} = combinedims(g.(chebgrid))
    G_decomposed::ttensor = hosvd(G; kwargs...)

    C::Array{Float64, m + 1} = G_decomposed.cten
    Us::Vector{Array{Float64, 2}} = G_decomposed.fmat

    # ghat(x, y, z) = C_d^abc u1_a(x) u2_b(y) u3_c(z) U4^l_d
    us::Vector{Array{Chebfun, 2}} = Vector{Array{Chebfun, 2}}(undef, m)
    for i in 1:m
        us[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
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
