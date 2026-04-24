# Charge basis

A charge basis represents a continuous U(1) degree of freedom ``e^\{i\varphi\}``
in its fourier (irrep) basis ``|n\rangle``, where ``n`` is an integer.
It may be used to represent quantum objects with integer charge, such as 
superconducting circuit elements.

The charge basis is a natural representation for circuit-QED elements such as
the "transmon", which has a hamiltonian of the form
```@example
using QuantumOptics # hide
ncut = 20
E_C = 1.0
E_J = 10.0
n_g = 0.0
b = ChargeBasis(ncut)
H = 4E_C * (n_g * identityoperator(b) + chargeop(b))^2 - E_J * cosφ(b)
nothing # hide
```
with energies periodic in the charge offset `n_g`.
See e.g. https://arxiv.org/abs/2005.12667.

**QuantumOptics.jl** provides the [`ChargeBasis`](@ref) and [`ShiftedChargeBasis`](@ref)
basis types, representing a truncated (and optionally shifted) fourier basis, with
`ncut` (or `nmin` and `nmax` in the shifted case) specifying the range of
available fourier modes.

## [States](@id charge: States)

* [`chargestate`](@ref)

## [Operators](@id charge: Operators)

* [`chargeop`](@ref)
* [`expiφ`](@ref)
* [`cosφ`](@ref)
* [`sinφ`](@ref)
