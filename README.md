# Approximating maps between linear spaces

Approximate functions of type
$$\reals^m \to \reals^n.$$

This package exists mostly so that I can use it in `ManiFactor.jl`.


## Example: Dominant singular value

Approximate
$$\mathrm{dominant~singular}(8 A_0 + x_1 A_1 + \dots + x_4 A_4),$$
where $A_0$, $\dots$, $A_4$ are randomly chosen $40 \times 60$ matrices such that $A_0$ is rank 1.
This figure illustrates the approximation error for different $N$ (nbr of sample points along each $x_i$):

![Plot](examples/dominant_singular_value.pdf)

TODO: cite the article for the error bound.
