import LinearAlgebra:
    normalize

# Just some quality of life functions


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
