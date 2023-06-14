# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
include("../Approximations.jl")
include("../QOL.jl")
using Plots, LinearAlgebra, Random, Combinatorics, Transducers

Lambda(N) = (2 / pi) * log(N + 1) + 1

# Smooth
function inverse_quadratic(;#={{{=#
    m=4,
    Ns=4:4:40,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    function g(x::Vector{Float64})::Float64
        return 1.0 / (1.0 + sum([xi^2 for xi in x]))
    end
    
    # This bound for V might not be very tight
    V(nu) = (1 + sqrt(m)) * factorial(big(nu))
    
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error")
    if savefigure; savefig("inverse_quadratic.pdf"); end
    display(p)
end#=}}}=#

# Smooth
function gaussian(;#={{{=#
    m=4,
    Ns=2:2:26,
    verbose=false,
    savefigure=false,
    kwargs...
    )
    function g(x::Vector{Float64})::Float64
        return exp(-sum([xi^2 for xi in x]))
    end
    
    V(nu) = 2 * sum([
        factorial(big(nu)) * big(2)^(nu - 2 * j) * exp(m) / (factorial(big(j)) * factorial(big(nu - 2 * j)))
        for j in 0:Int(floor(nu / 2))])
    
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error")
    if savefigure; savefig("gaussian.pdf"); end
    display(p)
end#=}}}=#

# C^1
function gaussian_modified(;#={{{=#
    m=4,
    Ns=2:2:26,
    verbose=false,
    savefigure=false,
    kwargs...
    )
    function g(x::Vector{Float64})::Float64
        return exp(-sum([sign(xi) * xi^2 for xi in x]))
    end
    
    V2 = 4 * exp(m)
    
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = 4 * V2 * (Lambda(N)^m - 1) / (pi * 2 * big(N - 2)^2 * (Lambda(N) - 1))
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        # ylims=(1e-16, 2 * maximum([es..., bs...])),
        # yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error")
    if savefigure; savefig("gaussian_modified.pdf"); end
    display(p)
end#=}}}=#

# Smooth
# Use tol=1e-15 to unlock full precision
function dominant_singular_value(;#={{{=#
    m=4,
    n1=40,
    n2=60,
    Ns=2:1:12,
    verbose=false,
    savefigure=false,
    figkwargs=(windowsize=(240, 160), guidefontsize=5, xtickfontsize=5, ytickfontsize=5, legendfontsize=5),
    kwargs...
    )

    Random.seed!(420)
    a = LinearAlgebra.normalize(rand(n1))
    b = LinearAlgebra.normalize(rand(n2))
    As = [LinearAlgebra.normalize(rand(n1, n2)) for _ in 1:m]
    function g(x) # : [-1, 1]^m -> Segre((m1, m2))
        U, S, Vt = svd(
            sum([xi * A for (xi, A) in zip(x, As)]) + 
            2 * m * a * b'
            )
        return S[1]
    end

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
        6300 * ginv1(l)^2 * ginv2(l)^2 * g3(l)^2 -
        280 * g1(l)^3 * g3(l)^3
        ))
    V8(l) = 2 * abs(1 / g1(l)^15 * (
        -135135 * g2(l)^7 +
        270270 * g1(l) * g2(l)^5 * g3(l) -
        138600 * g1(l)^2 * g2(l)^3 * g3(l)^2 +
        15400 * g1(l)^3 * g2(l) * g3(l)^3
        ))
    V9(l) = 2 * abs( 1 / g1(l)^17 * (
        2027025 * g2(l)^8 -
        4729725 * g1(l) * g2(l)^6 * g3(l) +
        3153150 * g1(l)^2 * g2(l)^4 * g3(l)^2 -
        600600 * g1(l)^3 * g2(l)^2 * g3(l)^3 + 
        15400 * g1(l)^4 * g3(l)^4
        ))
    V10(l) = 2 * abs(1 / g1(l)^19 * (
        -34459425 * g2(l)^9 +
        91891800 * g1(l) * g2(l)^7 * g3(l) -
        75675600 * g1(l)^2 * g2(l)^5 * g3(l)^2 +
        21021000 * g1(l)^3 * g2(l)^3 * g3(l)^3 -
        1401400 * g1(l)^4 * g2(l) * g3(l)^4
        ))#=}}}=#

    V(nu) = maximum([[V1, V2, V3, V3, V4, V5, V6, V7, V8, V9, V10, repeat([t -> NaN], 100)...][nu](l) for l in 1:m])
    
    global ghat
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])

        ghat = approximate_scalar(m, g; res=N, kwargs...)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])

        if verbose
            println("error ", es[i])
            x = ones(m) - 2.0 * rand(m)
            print("evaluating g     ")
            @time(g(x))
            print("evaluating ghat  ")
            @time(ghat(x))
            println()
        end
    end

    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        figkwargs...
        )
    scatter!(Ns, es;
        label="measured error",
        color=2
        )
    if savefigure; savefig("dominant_singular_value.pdf"); end
    display(p)
end#=}}}=#

### Example functions [-1, 1]^d -> R ###
# See https://en.wikipedia.org/wiki/Test_functions_for_optimization,
# and https://arxiv.org/pdf/1308.4008.pdf.
# Used in, for example, https://arxiv.org/pdf/2208.03380.pdf, https://arxiv.org/pdf/2211.11338.pdf

# Only C^0 so no guarantees for convergence
function ackley(;#={{{=#
    m=4,
    Ns=2:2:30,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    function g(xs_::Vector{Float64})::Float64
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

    es = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(;
        label="error bound",
        legend=:bottomright,
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2
        )
    if savefigure; savefig("ackley.pdf"); end
    display(p)
end#=}}}=#

# TODO: This is not C^1...
# function alpine(;#={{{=#
#     m=3,
#     Ns=20:20:200,
#     verbose=false,
#     savefigure=false,
#     kwargs...
#     )

#     V1 = 672.614 # integral of abs( d^2/dx^2 abs(x * 10 * sin(x * 10) + 0.1 * x * 10)) from x=-1 to x=1

#     function g(xs_)
#         d = length(xs_)
#         xs = xs_ * 10.0

#         return sum([abs(x * sin(x) + 0.1 * x) for x in xs])
#     end

#     es = [NaN for _ in Ns]
#     bs = [NaN for _ in Ns]
#     for (i, N) = enumerate(Ns)
#         if verbose; println(i, "/", length(Ns)); end

#         ghat = approximate_scalar(m, g; res=N, kwargs...)

#         bs[i] = 4 * V1 * (Lambda(N)^m - 1) / (pi * 1 * (N - 1)^1 * (Lambda(N) - 1))
#         # TODO: calculate max betterly
#         es[i] = maximum([
#             abs((g - ghat)(2 * rand(m) .- 1.0))
#             for _ in 1:1000])
#         if verbose; println("error ", es[i]); end
#     end
    
#     p = plot(Ns, bs;
#         label="error bound",
#         xlabel="N",
#         xticks=Ns,
#         yaxis=:log,
#         # ylims=(1e-16, 2 * maximum([es...])),
#         # yticks=([1e0, 1e-5, 1e-10, 1e-15]),
#         )
#     scatter!(Ns, es;
#         label="measured error",
#         color=2
#         )
#     if savefigure; savefig("alpine.pdf"); end
#     display(p)
# end#=}}}=#

# Smooth, so converges exponentially
function rastrigin(;#={{{=#
    m=3,
    Ns=10:10:90,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    V(nu) = 2 * 32.1699^(nu + 1)

    function g(xs_)
        d = length(xs_)
        xs = xs_ * 5.12

        A = 10.0
    
        return A * d + sum([xs[i]^2 - A * cos(2 * pi * xs[i]) for i in 1:d])
    end

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("rastrigin.pdf"); end
    display(p)
end#=}}}=#

# Polynomial of degree 4, so is exact after N = 4
function dixon(;#={{{=#
    m=4,
    Ns=2:2:20,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    # V(nu) =

    function g(xs_)
        d = length(xs_)
        xs = xs_ * 10.0
    
        return (xs[1] - 1)^2 + sum([i * (2 * xs[i]^2 - xs[i - 1])^2 for i in 2:d])
    end

    es = [NaN for _ in Ns]
    # bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        # bs[i] = minimum([
        #     4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        #     for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("dixon.pdf"); end
    display(p)
end#=}}}=#

# Smooth but with small characteristic wavelength
function griewank(;#={{{=#
    m=2,
    Ns=100:50:800,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    V(nu) = (600.0)^(nu + 1)

    function g(xs_)
        d = length(xs_)
        xs = xs_ * 600.0
    
        return (
            sum([x^2 / 4000.0 for x in xs])
            - prod([cos(xs[i] / sqrt(i)) for i in 1:d])
            + 1
            )
    end

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        # xaxis=:log,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("griewank.pdf"); end
    display(p)
end#=}}}=#

# Smooth
function schaffer(;#={{{=#
    m=3,
    Ns=100:20:320,
    verbose=false,
    savefigure=false,
    kwargs...
    )

    # V(nu) = TODO

    function g(xs_)
        d = length(xs_)
        xs = xs_ * 100.0

        return sum([
            0.5 + (sin(sqrt(xs[i]^2 + xs[i + 1]^2))^2 - 0.5) / (1.0 + 0.001 * (xs[i] + xs[i + 1]^2))^2
            for i in 1:(d - 1)])
    end

    es = [NaN for _ in Ns]
    # bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        # bs[i] = minimum([
        #     4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
        #     for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        # xaxis=:log,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("schaffer.pdf"); end
    display(p)
end#=}}}=#

# Smooth
# Computing the bound for this one is expensive as well
using TaylorSeries, Trapz
function michalewicz(;#={{{=#
    m=3,
    Ns=20:20:200,
    verbose=false,
    savefigure=false,
    kwargs...
    )

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

    function g(xs_)
        d = length(xs_)
        xs = (xs_ .+ 1) * pi / 2

        return -sum([sin(xs[i]) * sin(i * xs[i]^2 / pi)^(2 * M) for i in 1:d])
    end

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("michalewicz.pdf"); end
    display(p)
end#=}}}=#

# TODO: test_approximate_vector
function smooth_vector_field(;#={{{=#
    m=4,
    n=4,
    Ns=2:2:24,
    verbose=false,
    savefigure=false,
    kwargs...
    )


    A = rand(m, n)
    V(nu) = 2 * maximum(A * [i^(nu + 1) for i in 1:m])
    function g(xs)
    
        return A * [cos(i * x) for (i, x) in enumerate(xs)]
    end

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_vector(m, n, g; res=N, kwargs...)

        # TODO: calculate max betterly
        es[i] = maximum([
            maximum(abs.((g - ghat)(2 * rand(m) .- 1.0)))
            for _ in 1:1000])
        if verbose; println("error ", es[i]); end
    end
    
    p = plot(Ns, bs;
        label="error bound",
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum([es..., bs...])),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2)
    if savefigure; savefig("rastrigin_gradient.pdf"); end
    display(p)
end#=}}}=#
    
