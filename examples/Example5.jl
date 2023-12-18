# This example takes a while to run
using ApproximatingMapsBetweenLinearSpaces
using Plots; pyplot()
using Random; Random.seed!(1)

m=4
Ns=10:5:75

# Rastrigin
A = 10.0
function g(x_)#={{{=#
    x = x_ * 5.12

    return m * A + sum([x[i]^2 - A * cos(2 * pi * x[i]) for i in 1:m])
end#=}}}=#

Lambda(N) = (2 / pi) * log(N + 1) + 1 # Chebyshev interpolation operator norm
b(N) = min(Lambda(N)^m * (m * A + m * 5.12^2 + m * A),
    minimum([ # Bound for |g - ghat|
    let
        rho = beta + sqrt(beta^2 + 1)
        C = m * A + (m - 1) * (5.12^2 + A) + A * cosh(2 * pi * 5.12 * beta)
        4 * (Lambda(N) - 1) * C / ((rho - 1) * rho^N * (Lambda(N) - 1)) # TODO: reference the correct equation in the article
    end
    for beta in 1.0:0.2:10.0]))

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
plot!(p, Ns[1:end-2], bs[1:end-2]; label="error bound")
scatter!(p, Ns, es; label="measured error")

# To save figure and data to file:
using CSV
using DataFrames: DataFrame
savefig("Example5.png")
CSV.write("Example5.csv", DataFrame([:Ns => Ns, :es => es, :bs => bs]))
