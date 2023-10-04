using ApproximatingMapsBetweenLinearSpaces
using Plots; pyplot()

m=4
Ns=10:10:80

# Rastrigin
function g(x_)#={{{=#
    d = length(x_)
    x = x_ * 5.12

    A = 10.0

    return A * d + sum([x[i]^2 - A * cos(2 * pi * x[i]) for i in 1:d])
end#=}}}=#

V(nu) = 2 * 32.1699^(nu + 1) # Bound for |(d/dxi)^n g(x)|
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
savefig("Example5.png")
CSV.write("Example5.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
