# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
# TODO: Move some of these methods into Tests.jl?
include("../Approximations.jl")
using Plots
using LinearAlgebra
using Random

Lambda(N) = (2 / pi) * log(N + 1) + 1

function inverse_quadratic(;#={{{=#
    m=4,
    Ns=2:2:36,
    verbose=false,
    savefigure=false,
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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)
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

function gaussian(;#={{{=#
    m=4,
    Ns=2:2:30,
    verbose=false,
    savefigure=false,
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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)
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

function dominant_singular_value(;#={{{=#
    m=4,
    n1 = 20,
    n2 = 30,
    Ns=2:1:12,
    verbose=false,
    savefigure=false,
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
    
    es = [NaN for _ in Ns]
    global ghat # TODO: try local?
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])

        if verbose; println("error ", es[i]); end
    end

    # x = rand(m)
    # @time g(x)
    # @time ghat(x)
    
    p = plot(;
        xlabel="N",
        xticks=Ns,
        yaxis=:log,
        ylims=(1e-16, 2 * maximum(es)),
        yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2
        )
    if savefigure; savefig("dominant_singular_value.pdf"); end
    display(p)
end#=}}}=#

# Optimization benchmark functions
using Combinatorics, Transducers

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

        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

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

# C^1 so converges linearly
function alpine(;#={{{=#
    m=3,
    Ns=10:10:200,
    verbose=false,
    savefigure=false,
    )

    V1 = 672.614 # integral of abs( d^2/dx^2 abs(x * 10 * sin(x * 10) + 0.1 * x * 10)) from x=-1 to x=1

    function g(xs_)
        d = length(xs_)
        xs = xs_ * 10.0

        return sum([abs(x * sin(x) + 0.1 * x) for x in xs])
    end

    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

        bs[i] = 4 * V1 * (Lambda(N)^m - 1) / (pi * 1 * (N - 1)^1 * (Lambda(N) - 1))
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
        # ylims=(1e-16, 2 * maximum([es...])),
        # yticks=([1e0, 1e-5, 1e-10, 1e-15]),
        )
    scatter!(Ns, es;
        label="measured error",
        color=2
        )
    if savefigure; savefig("alpine.pdf"); end
    display(p)
end#=}}}=#

# Smooth, so converges exponentially
function rastrigin(;#={{{=#
    m=3,
    Ns=10:10:90,
    verbose=false,
    savefigure=false,
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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

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
# Computing the bound for this one is expensive as well
using TaylorSeries, Trapz
function michalewicz(;#={{{=#
    m=3,
    Ns=20:20:200,
    verbose=false,
    savefigure=false,
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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

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

# Smooth
# function schaffer(;#={{{=#
#     m=3,
#     Ns=100:20:320,
#     verbose=false,
#     )

#     # V(nu) =

#     function g(xs_)
#         d = length(xs_)
#         xs = xs_ * 100.0

#         return sum([
#             0.5 + (sin(sqrt(xs[i]^2 + xs[i + 1]^2))^2 - 0.5) / (1.0 + 0.001 * (xs[i] + xs[i + 1]^2))^2
#             for i in 1:(d - 1)])
#     end

#     es = [NaN for _ in Ns]
#     # bs = [NaN for _ in Ns]
#     for (i, N) = enumerate(Ns)
#         if verbose; println(i, "/", length(Ns)); end

#         # bs[i] = minimum([
#         #     4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
#         #     for nu in 1:(N - 1)])
    
#         ghat = approximate_scalar(m, g; res=N, complete_sampling=true)

#         # TODO: calculate max betterly
#         es[i] = maximum([
#             abs((g - ghat)(2 * rand(m) .- 1.0))
#             for _ in 1:1000])
#         if verbose; println("error ", es[i]); end
#     end
    
#     plot(;
#         label="error bound",
#         xlabel="N",
#         xticks=Ns,
#         # xaxis=:log,
#         yaxis=:log,
#         ylims=(1e-16, 2 * maximum([es...])),
#         yticks=([1e0, 1e-5, 1e-10, 1e-15]),
#         )
#     scatter!(Ns, es;
#         label="measured error",
#         color=2)
#     # savefig(".pdf")
# end#=}}}=#

# TODO: test approximate_scalar(..., complete_sampling=true) against predicted error bounds

# """ Testing that approximate_scalar is a good fit """
# function test_approximate_scalar(#={{{=#
#     ;verbose=false,
#     kwargs...
#     )

#     # TODO: when m = 1, do a normal chebfun
#     for m in 2:4
#         if verbose
#             println()
#             println("m = ", m)
#         end
#         for g in gs

#             ghat = approximate_scalar(m, g; kwargs...)
    
#             max_error = 0.0
#             x_max = zeros(m)
#             for _ in 1:10
#                 x = rand(m)
#                 g_x = g(x)
#                 ghat_x = ghat(x)
#                 error = abs(g_x - ghat_x)
#                 if error > max_error
#                     x_max = x
#                     max_error = error
#                 end
#             end

#             # if (error / abs(g_x) > 1e-10);
#             #     throw("approximate_scalar1 not accurate enough");
#             # end
#             if verbose
#                 println(rpad(g, 12, " "), " has relative error ", round(max_error / abs(g(x_max)); sigdigits=2))
#             end
#         end
#     end
# end#=}}}=#

# TODO:  test_approximate_vector

