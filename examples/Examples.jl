# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
# TODO: refactor to look more like ManiFactor.jl?
using ApproximatingMapsBetweenLinearSpaces
using LinearAlgebra, Random, Combinatorics, Transducers, Plots, CSV
using DataFrames: DataFrame
using TaylorSeries, Trapz # For michalewicz
include("../QOL.jl")

# TODO: How to save metadata like m and decomposition_method?

Lambda(N) = (2 / pi) * log(N + 1) + 1

function plot_errors(#={{{=#
    g::Function, # :: R^m -> R
    b::Function, # :: N -> R Error bound
    m::Int64,
    Ns;
    verbose=false,
    save=false,
    get_univariate_scheme=chebfun, # Int -> UnivariateApproximationScheme
    kwargs...
    )

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end
    
        ghat = approximate_scalar(
            m,
            g;
            univariate_scheme=get_univariate_scheme(N),
            kwargs...)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        bs[i] = b(N)
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        legend=:topright,
        )
    scatter!(Ns, es;
        label="measured error")
    if save
        savefig(string(g, ".pdf"))
        CSV.write(string(g, ".csv"), DataFrame([:Ns => Ns, :es => es, :bs => bs]))
    end
    display(p)
end#=}}}=#

# Smooth
# TODO: Pass the relevant keywords explicitly like verbose, save, etc?
# TODO: What are are all the relevant keywords?
"""
Approximates g(x) = 1 / (1 + x1^2 + ... + xm^2).
"""
function inverse_quadratic(;#={{{=#
    m=4,
    Ns=4:4:44,
    kwargs...
    )

    function inverse_quadratic(x::Vector{Float64})::Float64
        return 1.0 / (1.0 + sum([xi^2 for xi in x]))
    end
    
    # Error bound for tensorized Chebyshev interpolation. Might not be very tight.
    V(nu) = (1 + sqrt(m)) * factorial(big(nu))
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])
    # TODO: How can this grow for small N??
    
    plot_errors(
        inverse_quadratic,
        b,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

"""
Approximates
    g(x) = largest singular value of (A0 + x1 * A1 + ... + xm * Am),
where the A's are randomized n1 x n2 matrices chosen so that g is smooth.
"""
function dominant_singular_value(;#={{{=#
    m=4,
    n1=40,
    n2=60,
    Ns=2:1:12,
    kwargs...
    )

    Random.seed!(420)
    A0 = LinearAlgebra.normalize(rand(n1)) * LinearAlgebra.normalize(rand(n2))'
    As = [LinearAlgebra.normalize(rand(n1, n2)) for _ in 1:m]
    function dominant_singular_value(x) # : [-1, 1]^m -> Segre((m1, m2))
        U, S, Vt = svd(
            2 * m * A0 + 
            sum([xi * A for (xi, A) in zip(x, As)])
            )
        return S[1]
    end

    # nth derivative of the svd along axis l #={{{=#
    # https://www.jstor.org/stable/2695472
    g = dominant_singular_value
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

    # Error bound for tensorized Chebyshev interpolation.
    V(nu) = maximum([[V1, V2, V3, V3, V4, V5, V6, V7, V8, V9, V10, repeat([t -> NaN], 100)...][nu](l) for l in 1:m])
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])
    
    plot_errors(
        dominant_singular_value,
        b,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

# Smooth
function gaussian(;#={{{=#
    m=4,
    Ns=collect(2:2:26),
    kwargs...
    )

    function gaussian(x::Vector{Float64})::Float64
        return exp(-sum([xi^2 for xi in x]))
    end
    
    # Error bound for tensorized Chebyshev interpolation.
    V(nu) = 2 * sum([
        factorial(big(nu)) * big(2)^(nu - 2 * j) * exp(m) / (factorial(big(j)) * factorial(big(nu - 2 * j)))
        for j in 0:Int(floor(nu / 2))])
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])
    
    plot_errors(
        gaussian,
        b,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

# C^1
function gaussian_modified(;#={{{=#
    m=4,
    Ns=2:2:26,
    kwargs...
    )

    function gaussian_modified(x::Vector{Float64})::Float64
        return exp(-sum([sign(xi) * xi^2 for xi in x]))
    end
    
    # Error bound for tensorized Chebyshev interpolation.
    V2 = 4 * exp(m)
    b(N) = 4 * V2 * (Lambda(N)^m - 1) / (pi * 2 * big(N - 2)^2 * (Lambda(N) - 1))
    
    plot_errors(
        gaussian_modified,
        b,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

### Example functions [-1, 1]^d -> R ###
# See https://en.wikipedia.org/wiki/Test_functions_for_optimization,
# and https://arxiv.org/pdf/1308.4008.pdf.
# Used in, for example, https://arxiv.org/pdf/2208.03380.pdf, https://arxiv.org/pdf/2211.11338.pdf

# Only C^0 so no guarantees for convergence
function ackley(;#={{{=#
    m=4,
    Ns=2:2:30,
    kwargs...
    )

    function ackley(xs_::Vector{Float64})::Float64
        d = length(xs_)
        xs = xs_ * 32.768 # rescale so that x \in [-1, 1]^d
    
        A = 20.0
        B = 0.2
        C = 2 * pi
    
        return (
            -A * exp(-B * sqrt((1.0 / d) * sum([x^2 for x in xs])))
            - exp((1.0 / d) * sum([cos(C *x) for x in xs]))
            + A + exp(1)
            )
    end

    plot_errors(
        ackley,
        x -> NaN,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

# Smooth, so converges exponentially
function rastrigin(;#={{{=#
    m=4,
    Ns=10:10:80,
    kwargs...
    )

    # Error bound for tensorized Chebyshev interpolation.
    V(nu) = 2 * 32.1699^(nu + 1)
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])

    function rastrigin(xs_)
        d = length(xs_)
        xs = xs_ * 5.12

        A = 10.0
    
        return A * d + sum([xs[i]^2 - A * cos(2 * pi * xs[i]) for i in 1:d])
    end

    plot_errors(
        rastrigin,
        b,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

# Polynomial of degree 4, so is exact after N = 4
function dixon(;#={{{=#
    m=4,
    Ns=2:2:20,
    kwargs...
    )

    function dixon(xs_)
        d = length(xs_)
        xs = xs_ * 10.0
    
        return (xs[1] - 1)^2 + sum([i * (2 * xs[i]^2 - xs[i - 1])^2 for i in 2:d])
    end

    plot_errors(
        dixon,
        x -> NaN,
        m,
        Ns;
        kwargs...
        )
end#=}}}=#

# Smooth but with small characteristic wavelength
function griewank(;#={{{=#
    m=4,
    Ns=100:50:800,
    kwargs...
    )

    # Error bound for tensorized Chebyshev interpolation.
    V(nu) = (600.0)^(nu + 1)
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])

    function griewank(xs_)
        d = length(xs_)
        xs = xs_ * 600.0
    
        return (
            sum([x^2 / 4000.0 for x in xs])
            - prod([cos(xs[i] / sqrt(i)) for i in 1:d])
            + 1
            )
    end

    plot_errors(
        griewank,
        b,
        m,
        Ns;
        decomposition_method=TTsvd_incomplete, # Works fine for this function
        kwargs...
        )
end#=}}}=#

# Smooth
# Computing the bound for this one is expensive as well
function michalewicz(;#={{{=#
    m=4,
    Ns=20:20:240,
    kwargs...
    )

    function michalewicz(xs_)
        d = length(xs_)
        xs = (xs_ .+ 1) * pi / 2

        return -sum([sin(xs[i]) * sin(i * xs[i]^2 / pi)^(2 * M) for i in 1:d])
    end

    M = 10
    nan2zero(x) = isnan(x) ? 0.0 : x
    function V(nu)
        function integrand(x)
            t = Taylor1(Ns[end] + 1) + x
            T = sin((t + 1) * pi / 2) * sin(m * ((t + 1) * pi / 2)^2 / pi)^(2 * M)
            return nan2zero(abs(2 * getcoeff(T, nu + 1) * factorial(big(nu))))
        end

        i = trapz(range(-0.999, 0.999, length=200), integrand.(range(-0.999, 0.999, length=200)))
        
        return i
    end
    b(N) = minimum([
        4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        for nu in 1:(N - 1)])

    plot_errors(
        michalewicz,
        b,
        m,
        Ns;
        decomposition_method=TTsvd_incomplete, # Works fine for this function
        kwargs...
        )
end#=}}}=#

# Schwarz's theorem counterexample
# TODO: Error bounds

# TODO: test_approximate_vector
