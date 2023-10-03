import LinearAlgebra:
    normalize

# Just some quality of life functions

""" Partial application """
function pa(f, a...; pos=1)#={{{=#
    return (b...) -> f([b...][1:(pos - 1)]..., a..., [b...][(pos + length([a...]) - 1):end]...)
end#=}}}=#

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

import Base.+#={{{=#
import Base.-
function +(
    a::Function,
    b::Function
    )::Function

    return t -> a(t) + b(t)
end
function -(
    a::Function,
    b::Function
    )::Function

    return t -> a(t) - b(t)
end
function -(
    a::Function
    )::Function

    return t -> -a(t)
end#=}}}=#
