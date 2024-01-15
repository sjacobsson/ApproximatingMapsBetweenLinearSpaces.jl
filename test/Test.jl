# Run Example 1 with different combinations of parameters
using ApproximatingMapsBetweenLinearSpaces
using TensorToolbox: sthosvd, hosvd, TTsvd, cp_als

# Verbose isapprox
import Base.isapprox
function isapprox(a, b, verbose; kwargs...)#={{{=#
    if verbose; println(a, " ≈ ", b); end

    return isapprox(a, b; kwargs...)
end#=}}}=#

function g(x)
    return 1.0 / (1.0 + sum([xi^2 for xi in x]))
end

function main(;#={{{=#
    ms=2:4,
    Ns=10:20,
    nbr_tests=10,
    decomposition_methods=[sthosvd, hosvd, TTsvd], # TODO: also test cp_als, right now it sometimes fails with SinSingularException
    verbose=false,
    )

    for m in ms
        if verbose; println("m = ", m); end

        Lambda(N) = (2 / pi) * log(N + 1) + 1 # Chebyshev interpolation operator norm
        b(N) = minimum([ # Bound for |g - ghat|
            let
                rho = beta + sqrt(beta^2 + 1)
                C = 1 / (1 - beta^2)
                4 * (Lambda(N)^m - 1) * C / ((rho - 1) * rho^N * (Lambda(N) - 1))
            end
            for beta in 0.0:0.1:1.0])

        for N in Ns
            if verbose; println("N = ", N); end

            b_N = b(N)
            if verbose; println("bound = ", b_N); end

            for decomposition_method in decomposition_methods
                if verbose; println("decomposition_method = ", decomposition_method); end
            
                local ghat = approximate_scalar(
                    m,
                    g;
                    decomposition_method=decomposition_method,
                    univariate_scheme=chebyshev(N),
                    tolerance=b_N / 100,
                    )

                for _ in 1:nbr_tests
                    x = 2 * rand(m) .- 1.0
                    @assert(isapprox(g(x), ghat(x), verbose; atol=b_N))
                end
                if verbose; println(" "); end
            end
        end
    end
end#=}}}=#
