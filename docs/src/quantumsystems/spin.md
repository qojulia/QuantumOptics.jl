# Spin

```@example
using QuantumOptics # hide
b = SpinBasis(3//2)
psi = spinup(b)
sx = sigmax(b)
nothing # hide
```

Spin systems of arbitrary spin number can be modeled with the [`SpinBasis`](@ref) which is defined as:

```julia
struct SpinBasis <: Basis
    shape::Vector{Int}
    spinnumber::Rational{Int}
end
```


## [States](@id spin: States)

The lowest and uppermost states are defined:

* [`spinup`](@ref)
* [`spindown`](@ref)


## [Operators](@id spin: Operators)

All expected operators are implemented, all of which require a single argument of the type [`SpinBasis`](@ref).

* [`sigmax`](@ref)
* [`sigmay`](@ref)
* [`sigmaz`](@ref)
* [`sigmap`](@ref)
* [`sigmam`](@ref)


## [Examples](@id spin: Examples)

* [Jaynes-Cummings model](@ref)
* [Entanglement of Two Qubits](@ref)
