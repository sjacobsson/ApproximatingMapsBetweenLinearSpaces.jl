using ApproximatingMapsBetweenLinearSpaces
using LinearAlgebra
using Random
using Plots; pyplot()

m=4
n1=40
n2=60
Ns=2:1:12

Random.seed!(420)
A0 = normalize(rand(n1)) * normalize(rand(n2))'
As = [normalize(rand(n1, n2)) for _ in 1:m]
function g(x)
    U, S, Vt = svd(
        2 * m * A0 + 
        sum([xi * A for (xi, A) in zip(x, As)])
        )
    return S[1]
end

""" Approximate derivative of f at x """
function finite_difference(#={{{=#
    f::Function, # :: ℝ -> some vector space
    x::Float64,
    h::Float64;
    order=1::Int64
    )

    # https://en.wikipedia.org/wiki/Finite_difference_coefficient
    if order == 1
        return (
            (1 / 12) *  f(x - 2 * h) +
            (-2 / 3) *  f(x - 1 * h) +
            (2 / 3) *   f(x + 1 * h) +
            (-1 / 12) * f(x + 2 * h)
            ) / h
    elseif order == 2
    return (
        (-1 / 12) * f(x - 2 * h) +
        (4 / 3) *   f(x - 1 * h) +
        (-5 / 2) *  f(x) +
        (4 / 3) *   f(x + 1 * h) +
        (-1 / 12) * f(x + 2 * h)
        ) / h^2
    elseif order == 3
    return (
        (1 / 8) *   f(x - 3 * h) +
        (-1) *      f(x - 2 * h) +
        (13 / 8) *  f(x - 1 * h) +
        (-13 / 8) * f(x + 1 * h) +
        (1) *       f(x + 2 * h) +
        (-1 / 8) *  f(x + 3 * h)
        ) / h^3
    end
end#=}}}=#

# nth derivative of the svd along axis l #={{{=#
# https://www.jstor.org/stable/2695472
g1(l) = big(norm(finite_difference(t -> g([i == l ? t : 0 for i in 1:m]), 0.0, 1e-5, order=1)))
ginv1(l) = 1 / g1(l)
g2(l) = big(norm(finite_difference(t -> g([i == l ? t : 0 for i in 1:m]), 0.0, 1e-4, order=2)))
ginv2(l) = -g2(l) / g1(l)^3
g3(l) = big(norm(finite_difference(t -> g([i == l ? t : 0 for i in 1:m]), 0.0, 1e-3, order=3)))
ginv3(l) = 1 / g1(l)^5 * ( 3* g2(l)^2 - g1(l) * g3(l) )

V1(l) = 2 * abs(g1(l))
V2(l) = 2 * abs(g2(l))
V3(l) = 2 * abs(g3(l))
V4(l) = 2 * abs(1 / ginv1(l)^7 * (
    -15 * ginv2(l)^3 +
    10 * ginv1(l) * ginv2(l) * ginv3(l)
    ))
V5(l) = 2 * abs(1 / ginv1(l)^9 * (
    105 * ginv2(l)^4 -
    105 * ginv1(l) * ginv2(l)^2 * ginv3(l) +
    10 * ginv1(l)^2 * ginv3(l)^2
    ))
V6(l) = 2 * abs(1 / ginv1(l)^11 * (
    -945 * ginv2(l)^5 +
    1260 * ginv1(l) * ginv2(l)^3 * ginv3(l) -
    280 * ginv1(l)^2 * ginv2(l) * ginv3(l)^2
    ))
V7(l) = 2 * abs(1 / ginv1(l)^13 * (
    10395 * ginv2(l)^6 -
    17325 * ginv1(l) * ginv2(l)^4 * ginv3(l) +
    6300 * ginv1(l)^2 * ginv2(l)^2 * ginv3(l)^2 -
    280 * ginv1(l)^3 * ginv3(l)^3
    ))
V8(l) = 2 * abs(1 / ginv1(l)^15 * (
    -135135 * ginv2(l)^7 +
    270270 * ginv1(l) * ginv2(l)^5 * ginv3(l) -
    138600 * ginv1(l)^2 * ginv2(l)^3 * ginv3(l)^2 +
    15400 * ginv1(l)^3 * ginv2(l) * ginv3(l)^3
    ))
V9(l) = 2 * abs( 1 / ginv1(l)^17 * (
    2027025 * ginv2(l)^8 -
    4729725 * ginv1(l) * ginv2(l)^6 * ginv3(l) +
    3153150 * ginv1(l)^2 * ginv2(l)^4 * ginv3(l)^2 -
    600600 * ginv1(l)^3 * ginv2(l)^2 * ginv3(l)^3 + 
    15400 * ginv1(l)^4 * ginv3(l)^4
    ))
V10(l) = 2 * abs(1 / ginv1(l)^19 * (
    -34459425 * ginv2(l)^9 +
    91891800 * ginv1(l) * ginv2(l)^7 * ginv3(l) -
    75675600 * ginv1(l)^2 * ginv2(l)^5 * ginv3(l)^2 +
    21021000 * ginv1(l)^3 * ginv2(l)^3 * ginv3(l)^3 -
    1401400 * ginv1(l)^4 * ginv2(l) * ginv3(l)^4
    ))#=}}}=#

V(nu) = maximum([[V1, V2, V3, V3, V4, V5, V6, V7, V8, V9, V10, repeat([t -> NaN], 100)...][nu](l) for l in 1:m]) # Bound for |(d/dxi)^n g(x)|
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
        eps_rel=1e-15
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
savefig("Example2.png")
CSV.write("Example2.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
