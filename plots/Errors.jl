# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
# TODO: Move some of these methods into Tests.jl?
include("../Approximations.jl")
using Plots
using LinearAlgebra
using Random

Lambda(N) = (2 / pi) * log(N + 1) + 1

function inverse_quadratic(;#={{{=#
    verbose=false,
    Ns=10:2:40
    )

    m = 4
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
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        # ghat = approximate_scalar(m, g; res=N, complete_sampling=false, reqrank=Int(round(N/2)))
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:100])
    end
    
    plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    scatter!(Ns, es; label="measured error")
    # savefig("inverse_quadratic.pdf")
end#=}}}=#

function gaussian(;#={{{=#
    verbose=false,
    Ns=10:30
    )
    m = 4
    function g(x::Vector{Float64})::Float64
        return exp(sum([xi^2 for xi in x]))
    end
    
    V(nu) = 2 * sum([
        factorial(big(nu)) * big(2)^(nu - 2 * j) * exp(m) / (factorial(big(j)) * factorial(big(nu - 2 * j)))
        for j in 0:Int(floor(nu / 2))])
    
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        ghat = approximate_scalar(m, g; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        # ghat = approximate_scalar(m, g; res=N, complete_sampling=false, reqrank=2 * Int(round(sqrt(N))))
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:100])
    end
    
    plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    scatter!(Ns, es; label="measured error")
    # savefig("gaussian.pdf")
end#=}}}=#

function dominant_singular_value(;#={{{=#
    verbose=false,
    m=4,
    Ns=10:2:30
    )

    m1 = 20
    m2 = 30

    a = normalize(rand(m1))
    b = normalize(rand(m2))
    u1 = normalize(rand(m1))
    v1 = normalize(rand(m2))
    u2 = normalize(rand(m1))
    v2 = normalize(rand(m2))
    u3 = normalize(rand(m1))
    v3 = normalize(rand(m2))
    u4 = normalize(rand(m1))
    v4 = normalize(rand(m2))
    function g(x::Vector{Float64})::Float64
        U, S, Vt = svd(
            0.1 * x[1] * u1 * v1' +
            0.1 * x[2] * u2 * v2' +
            0.1 * x[3] * u3 * v3' + 
            0.1 * x[4] * u4 * v4' + 
            1 * a * b'
            )
        return S[1]
    end
    
    es = [NaN for _ in Ns]
    global ghat # TODO: try local?
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        ghat = approximate_scalar(m, g; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        # ghat = approximate_scalar(m, g; res=N, complete_sampling=false, reqrank=Int(round(N / 2)))
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:1000])
    end

    # x = rand(m)
    # @time g(x)
    # @time ghat(x)
    
    # TODO: Save data to file
    plot(; yaxis=:log, xlabel="N") # No error bounds yet
    scatter!(Ns, es; label="measured error", color=:2)
    # savefig("dominant_singular_value.pdf")
end#=}}}=#
