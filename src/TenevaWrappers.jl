# Wraps some tensor decomposition methods from python package teneva

using TensorToolbox
using PyCall: (pyimport); teneva = pyimport("teneva")

"""
    function TTsvd_incomplete(
        G::Function, # :: N x ... x N -> R
        valence::Vector{Int64};
        reqrank::Int64=10,
        kwargs...
        )::TTtensor

Approximate a TT decomposition of a tensor G without using the full tensor. G is hence represented as a map from m-tuples of integers to the reals.
"""
function TTsvd_incomplete(#={{{=#
    G::Function, # :: N x ... x N -> R
    valence::Vector{Int64};
    reqrank::Int64=10, # TODO: If I rewrite this, make reqrank a tuple of Ints
    kwargs...
    )::TTtensor

    Is, idx, idx_many = teneva.sample_tt(valence, r=reqrank)
    Gs = [G(I .+ 1) for I in eachrow(Is)]
    return TTtensor(teneva.svd_incomplete(Is, Gs, idx, idx_many, r=reqrank; kwargs...))
end#=}}}=#

"""
    function TTsvd_cross(
        G::Function, # :: N x ... x N -> R
        valence::Vector{Int64};
        reqrank::Int64=10,
        kwargs...
        )::TTtensor

Approximate a TT decomposition of a tensor G without using the full tensor. G is hence represented as a map from m-tuples of integers to the reals.
"""
function TTsvd_cross(#={{{=#
    G::Function, # :: N x ... x N -> R
    valence::Vector{Int64};
    reqrank::Int64=10,
    kwargs...
    )::TTtensor
    println(valence)
    println(G([1, 2, 3, 4, 5]))

    m = 1000 # Nbr calls to target function
    e = 1e-10 # Desired accuracy
    return TTtensor(teneva.cross(G, teneva.rand(valence, reqrank), m, e, kwargs...))
end#=}}}=#

