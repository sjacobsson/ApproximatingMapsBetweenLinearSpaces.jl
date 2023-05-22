include("Approximations.jl")

using Combinatorics, Transducers
using LinearAlgebra

### Example functions [-1, 1]^d -> R ### #={{{=#
# Compiled in https://arxiv.org/pdf/1308.4008.pdf
# Used in, for example, https://arxiv.org/pdf/2208.03380.pdf, https://arxiv.org/pdf/2211.11338.pdf

function ackley(xs_::AbstractVector)::AbstractFloat#={{{=#
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
end#=}}}=#

function alpine(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 10.0

    return sum([abs(x * sin(x) + 0.1 * x) for x in xs])
end#=}}}=#

function dixon(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 10.0

    return (xs[1] - 1)^2 + sum([i * (2 * xs[i]^2 - xs[i - 1])^2 for i in 2:d])
end#=}}}=#

function exponential(xs::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs)

    return -exp(-(1 / 2.0) * sum([x^2 for x in xs]))
end#=}}}=#

function grienwank(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 600.0

    return (
        sum([x^2 / 4000.0 for x in xs])
        - prod([cos(xs[i] / sqrt(i)) for i in 1:d])
        + 1
        )
end#=}}}=#

function michalewicz(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = (xs_ .+ 1) * pi / 2

    m = 10

    return -sum([sin(xs[i]) * sin(i * xs[i]^2 / pi)^(2 * m) for i in 1:d])
end#=}}}=#

function piston(xs::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs)
    @assert(d == 7)
    M   = (60.0 - 30.0) * xs[1] / 2     + (60.0 + 30.0) / 2
    S   = (0.020 - 0.005) * xs[2] / 2   + (0.020 + 0.005) / 2
    V0  = (0.010 - 0.005) * xs[3] / 2   + (0.010 + 0.005) / 2
    k   = (5e3 - 1e3) * xs[4] / 2       + (5e3 + 1e3) / 2
    P0  = (11e4 - 9e4) * xs[5] / 2      + (11e4 + 9e4) / 2
    Ta  = (296.0 - 290.0) * xs[6] / 2   + (296.0 + 290.0) / 2
    T0  = (360.0 - 340.0) * xs[7] / 2   + (360.0 + 340.0) / 2

    A = P0 * S + 19.62 * M - k * V0 / S
    V = S * (sqrt(A^2 + 4 * k * P0 * V0 * Ta / T0) - A) / (2 * k)

    return 2 * pi * sqrt(M / (k + S^2 * P0 * V0 * Ta / (T0 * V^2)))
end#=}}}=#

function qing(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = (xs_ .+ 1.0) * 250.0

    return sum([(xs[i]^2 - i)^2 for i in 1:d])
end#=}}}=#

function rastrigin(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 5.12

    A = 10.0

    return A * d + sum([xs[i]^2 - A * cos(2 * pi * xs[i]) for i in 1:d])
end#=}}}=#

function rosenbrock(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 2.048

    return sum([(100 * (xs[i + 1] - xs[i]^2)^2 + (1.0 - xs[i])^2) for i in 1:(d - 1)])
end#=}}}=#

function schaffer(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 100.0

    return sum([
        0.5 + (sin(sqrt(xs[i]^2 + xs[i + 1]^2))^2 - 0.5) / (1.0 + 0.001 * (xs[i] + xs[i + 1]^2))^2
        for i in 1:(d - 1)])
end#=}}}=#

function schwefel(xs_::AbstractVector)::AbstractFloat#={{{=#
    d = length(xs_)
    xs = xs_ * 500.0

    return 428.9829 * d - sum([x * sin(sqrt(abs(x))) for x in xs])
end#=}}}=#

gs = [
    ackley,
    alpine,
    dixon,
    exponential,
    grienwank, # TODO: nw \mapsto w
    michalewicz,
    # piston,
    qing,
    rastrigin,
    rosenbrock,
    schaffer,
    schwefel
    ]
#=}}}=#

# TODO: test approximate_scalar(..., complete_sampling=true) against predicted error bounds

""" Testing that approximate_scalar is a good fit """
function test_approximate_scalar(#={{{=#
    ;verbose=false,
    kwargs...
    )

    # TODO: when m = 1, do a normal chebfun
    for m in 2:5
        if verbose
            println()
            println("m = ", m)
        end
        for g in gs

            ghat = approximate_scalar(g, m; kwargs...)
    
            max_error = 0.0
            x_max = zeros(m)
            for _ in 1:10
                x = rand(m)
                g_x = g(x)
                ghat_x = ghat(x)
                error = abs(g_x - ghat_x)
                if error > max_error
                    x_max = x
                    max_error = error
                end
            end

            # if (error / abs(g_x) > 1e-10);
            #     throw("approximate_scalar1 not accurate enough");
            # end
            if verbose
                println(rpad(g, 12, " "), " has relative error ", round(max_error / abs(g(x_max)); sigdigits=2))
            end
        end
    end
end#=}}}=#

# TODO:  test_approximate_vector

