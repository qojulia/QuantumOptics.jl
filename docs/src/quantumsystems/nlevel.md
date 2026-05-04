# N-Level

```@example
using QuantumOptics # hide
N = 3
b = NLevelBasis(N)
psi1 = nlevelstate(b, 1)
psi2 = nlevelstate(b, 2)
@assert psi1 == transition(b, 1, 2) * psi2
```

N-Level systems are mostly used as idealized model when the relevant physics can be captured in a small subspace of the complete Hilbert space. For example an atom can often be represented by a few relevant levels.

For these kinds of systems the [`NLevelBasis`](@ref) can be used. Essentially it is defined just as:

```julia
struct NLevelBasis <: Basis
    shape::Vector{Int}
    N::Int
end
```


## [States](@id nlevel: States)

We can create a state ``|m\rangle`` with

* [`nlevelstate(b::NLevelBasis, m::Int)`](@ref)


## [Operators](@id nlevel: Operators)

With the transition operator, we can create projectors of the form ``|m\rangle\langle n|`` describing a transition from the state ``|n\rangle`` to ``|m\rangle``.

* [`transition(b::NLevelBasis, m::Int, n::Int)`](@ref)


## [Examples](@id nlevel: Examples)

* [Raman Transition of a ``\Lambda``-scheme Atom](@ref)
* [Four level system in many-body formalism](@ref)
