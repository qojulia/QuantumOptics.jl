# Master time evolution

The dynamics of open quantum systems are governed by a master equation in Lindblad form:

```math
\dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
             + \sum_i \big(
                    J_i \rho J_i^\dagger
                    - \frac{1}{2} J_i^\dagger J_i \rho
                    - \frac{1}{2} \rho J_i^\dagger J_i
                \big)
```

It is implemented by the function

```@example master
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
J = [destroy(b)] # hide
tspan = [0,0.1] # hide
psi0 = fockstate(b, 0) # hide
timeevolution.master(tspan, psi0, H, J)
nothing # hide
```

The arguments required are quite similar to the ones of [`timeevolution.schroedinger`](@ref). `tspan` is a vector of times, `rho0` the initial state and `H` the Hamiltonian. We now also need the vector `J` that specifies the jump operators of the system.

The additional arguments available are

* `rates::{Vector{Float64}, Matrix{Float64}}`
* `Jdagger::Vector`
* `fout::Function`

The first specifies the decay rates of the system with default values one. If `rates` is a vector of length `length(J)`, then the `i` th entry of `rates` is paired with the `i`-th entry of `J`, such that `J_i` decays with ``\gamma_i``. If, on the other hand, `rates` is a matrix, then all entries of `J` are paired with one another and matched with the corresponding entrie of `rates`, resulting
in a Lindblad term of the form ``\sum_{i,j}\gamma_{ij}\left(J_i\rho J_j^\dagger - J_i^\dagger J_j\rho/2 - \rho J_i^\dagger J_j/2\right)``.

The second keyword argument can be used to pass a specific set of jump operators to be used in place of all ``J^\dagger`` appearances in the Lindblad term.

We can pass an output function just like the one for a Schrödinger equation. Note, though, that now the function must be defined with the arguments `fout(t, rho)`.

For performance reasons the solver internally first creates the non-hermitian Hamiltonian ``H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i`` and solves the equation

```math
\dot{\rho} = -\frac{i}{\hbar} \big[H_\mathrm{nh},\rho\big]
             + \sum_i J_i \rho J_i^\dagger
```

If for any reason this behavior is unwanted, e.g. special operators are used that don't support addition, the function [`timeevolution.master_h`](@ref) (h for hermitian) can be used.


## [Functions](@id master: Functions)

* [`timeevolution.master`](@ref)
* [`timeevolution.master_h`](@ref)
* [`timeevolution.master_nh`](@ref)
* [`timeevolution.master_dynamic`](@ref)
* [`timeevolution.master_nh_dynamic`](@ref)


## [Examples](@id master: Examples)

* [Pumped cavity](@ref)
* [Jaynes-Cummings model](@ref)
* [Raman Transition of a ``\Lambda``-scheme Atom](@ref)
* [Four level system in many-body formalism](@ref)
