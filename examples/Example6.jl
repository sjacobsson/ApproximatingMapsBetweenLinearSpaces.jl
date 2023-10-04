using ApproximatingMapsBetweenLinearSpaces
using Plots; pyplot()

using TensorToolbox
import ApproximatingMapsBetweenLinearSpaces: approximate_scalar
using PyCall: (pyimport); teneva = pyimport("teneva")

function TTsvd_incomplete(#={{{=#
    G::Function, # :: N x ... x N -> R
    valence::Vector{Int64};
    reqrank::Int64=10, # TODO: If I rewrite this, make reqrank a tuple of Ints
    kwargs...
    )::TTtensor

    Is, idx, idx_many = teneva.sample_tt(valence, r=reqrank)
    Gs = [G(collect(I)) for I in eachrow(Is)]
    return TTtensor(teneva.svd_incomplete(Is, Gs, idx, idx_many, r=reqrank; kwargs...))
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

m=4
Ns=100:50:800

# Griewank
function g(x_)#={{{=#
    d = length(x_)
    x = x_ * 600.0
    
    return (
        sum([xi^2 / 4000.0 for xi in x])
        - prod([cos(x[i] / sqrt(i)) for i in 1:d])
        + 1
        )
end#=}}}=#

V(nu) = (600.0)^(nu + 1) # Bound for |(d/dxi)^n g(x)|
Lambda(N) = (2 / pi) * log(N + 1) + 1 # Chebyshev interpolation operator norm
b(N) = minimum([ # Bound for |g - ghat|
    4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
    for nu in 1:(N - 1)])

# Loop over nbr of interpolation points
es = [NaN for _ in Ns]
bs = [NaN for _ in Ns]
for (i, N) = enumerate(Ns)
    local ghat = approximate_scalar(
        m,
        g;
        univariate_scheme=chebfun(N),
        decomposition_method=TTsvd_incomplete,
        # eps_rel=1e-15
        )

    # e = max(|g - ghat|)
    es[i] = maximum([
        abs(g(x) - ghat(x))
        for x in [2 * rand(m) .- 1.0 for _ in 1:1000]])
    bs[i] = b(N)
end

p = plot(;
    xlabel="N",
    xticks=Ns,
    yaxis=:log,
    ylims=(1e-16, 2 * maximum([es..., bs...])),
    yticks=([1e0, 1e-5, 1e-10, 1e-15]),
    legend=:topright,
    )
plot!(p, Ns, bs; label="error bound")
scatter!(p, Ns, es; label="measured error")

# To save figure and data to file:
using CSV
using DataFrames: DataFrame
savefig("Example6.png")
CSV.write("Example6.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
