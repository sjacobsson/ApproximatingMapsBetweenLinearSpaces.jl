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

"""
    function approximate_scalar(
        m::Int64,
        g::Function; # :: [-1, 1]^m -> R
        res::Int64=20, # nbr of interpolation points in each direction
        complete_sampling::Bool=false,
        kwargs...
        )::Function

Approximate a scalar-valued function using approximate TT decomposition and Chebyshev interpolation
"""
function approximate_scalar(#={{{=#
    m::Int64,
    g::Function; # :: [-1, 1]^m -> R
    res::Int64=20, # nbr of interpolation points in each direction
    complete_sampling::Bool=true,
    kwargs...
    )::Function

    # Evaluate g on Chebyshev grid
    # G_ijk = g(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G_ijk = C1^a_ib C2^b_jc C3^c_ka
    local G_decomposed::TTtensor
    chebpts::Vector{Float64} = chebyshevpoints(res)
    if complete_sampling
        chebgrid = [chebpts[collect(I)] for I in product(repeat([1:res], m)...)]
        G::Array{Float64, m} = g.(chebgrid)
        G_decomposed = TTsvd(G; kwargs...)
    else
        G_(I::Vector{Int64})::Float64 = g([chebpts[i] for i in I])
        valence = repeat([res], m)
        G_decomposed = TTsvd_incomplete(G_, valence; kwargs...)
    end
    Cs::Vector{Array{Float64, 3}} = G_decomposed.cores

    # ghat(x, y, z) = c1^a_b(x) c2^b_c(y) c3^c_a(z)
    cs::Vector{Array{Chebfun, 3}} = Vector{Array{Chebfun, 3}}(undef, m)
    for i in 1:m
        cs[i] = mapslices(
            pa(Fun, Chebyshev()) ∘ pa(transform, Chebyshev()), # Interpolate
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

"""
    function approximate_vector(
        m::Int64,
        n::Int64,
        g::Function; # :: [-1, 1]^m -> R^n
        res::Int64=20, # nbr of interpolation points in each direction
        complete_sampling::Bool=true,
        kwargs...
        )::Function

Approximate a vector-valued function using approximate TT decomposition and Chebyshev interpolation
"""
function approximate_vector(#={{{=#
    m::Int64,
    n::Int64,
    g::Function; # :: [-1, 1]^m -> R^n
    res::Int64=20, # nbr of interpolation points in each direction
    complete_sampling::Bool=true,
    kwargs...
    )::Function

    @assert(length(g(zeros(m))) == n)
    valence = [n, repeat([res], m)...]

    # Evaluate g on Chebyshev grid
    # G^l_ijk = g^l(t_i, t_j, t_k)
    # where t_i is the i:th chebyshev node, then decompose
    # G^l_ijk = C1^al_b C2^b_ic C3^c_jd C4^d_ka
    local G_decomposed::TTtensor
    chebpts::Vector{Float64} = chebyshevpoints(res)
    if complete_sampling
        chebgrid = [chebpts[collect(I)] for I in product(repeat([1:res], m)...)]
        G::Array{Float64, m + 1} = combinedims(g.(chebgrid))
        G_decomposed = TTsvd(G; kwargs...)
    else
        G_(I::Vector{Int64})::Float64 = g([chebpts[i] for i in I[2:end]])[I[1]]
        G_decomposed = TTsvd_incomplete(G_, valence; kwargs...)
    end
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

    # TODO: something like this?
    # if verbose
    #     println("decomposed valence ", valence, " tensor with TTrank ", TTrank(G_decomposed), ".")
    # end

    return g_approx
end#=}}}=#
