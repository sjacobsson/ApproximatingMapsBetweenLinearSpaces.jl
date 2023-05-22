# Compute errors from approximating different functions and compare to the error
# bound predicted by thm TODO.
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
    
        ghat = approximate_scalar(g, m; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        # ghat = approximate_scalar(g, m; res=N, complete_sampling=false, reqrank=Int(round(N/2)))
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:100])
    end
    
    # TODO: Save data to file
    plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    scatter!(Ns, es; label="measured error")
end#=}}}=#

function gaussian()#={{{=#
    m = 4
    function g(x::Vector{Float64})::Float64
        return exp(sum([xi^2 for xi in x]))
    end
    
    V(nu) = 2 * sum([
        factorial(big(nu)) * big(2)^(nu - 2 * j) * exp(m) / (factorial(big(j)) * factorial(big(nu - 2 * j)))
        for j in 0:Int(floor(nu / 2))])
    
    Ns = 10:30
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    for (i, N) = enumerate(Ns)
        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        # ghat = approximate_scalar(g, m; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        ghat = approximate_scalar(g, m; res=N, complete_sampling=false, reqrank=2 * Int(round(sqrt(N))))
        # TODO: calculate max betterly
        es[i] = maximum([
            abs((g - ghat)(2 * rand(m) .- 1.0))
            for _ in 1:100])
    end
    
    # TODO: Save data to file
    plot(Ns, bs; yaxis=:log, label="error bound", xlabel="N")
    scatter!(Ns, es; label="measured error")
end#=}}}=#

function dominant_singular_value(;#={{{=#
    verbose=false,
    m1=2,
    m2=3,
    Ns=10:2:20
    )
    m = m1 * m2

    Random.seed!(123)
    a = rand(m1)
    b = rand(m2)

    function g(x::Vector{Float64})::Float64
        return svd(0.1 * reshape(x, m1, m2) + a * b' / (norm(a) * norm(b))).S[1]
    end
    
    V(nu) = 2 * sum([
        factorial(big(nu)) * big(2)^(nu - 2 * j) * exp(m) / (factorial(big(j)) * factorial(big(nu - 2 * j)))
        for j in 0:Int(floor(nu / 2))])
    
    es = [NaN for _ in Ns]
    bs = [NaN for _ in Ns]
    global ghat
    for (i, N) = enumerate(Ns)
        if verbose; println(i, "/", length(Ns)); end

        bs[i] = minimum([
            4 * V(nu) * (Lambda(N)^m - 1) / (pi * nu * big(N - nu)^nu * (Lambda(N) - 1))
            for nu in 1:(N - 1)])
    
        # ghat = approximate_scalar(g, m; res=N, complete_sampling=true, reqrank=[N for _ in 1:(m - 1)])
        ghat = approximate_scalar(g, m; res=N, complete_sampling=false, reqrank=Int(round(N / 2)))
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
end#=}}}=#
