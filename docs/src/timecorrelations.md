# Two-time correlation functions

We can calculate the two-time correlation function ``g(t) = \langle A(t)B\rangle`` of a system for a given list of times with the implemented function [`timecorrelations.correlation`](@ref)

```@example timecorrelations
using QuantumOptics # hide
basis = FockBasis(10) # hide
a = destroy(basis) # hide
A = a # hide
B = dagger(A) # hide
H = a + dagger(a) # hide
J = [a] # hide
tspan = [0:0.1:10;]
ρ0 = dm(fockstate(basis, 0)) # hide
g = timecorrelations.correlation(tspan, ρ0, H, J, A, B)
nothing # hide
```
The calculation is performed by applying the operator `B` to the initial density matrix `ρ0`, which is then used as initial state for a time evolution according to a master equation with [`timeevolution.master`](@ref). The correlation `g` is then just the expectation value of `A` at every time-step.

If the list of times `tspan` is omitted, the function automatically calculates the correlation until steady-state is reached.

Furthermore, we can calculate the spectrum by means of the Wiener-Khinchin theorem, i.e. as Fourier transform of the correlation function,

```math
S(\omega) = 2\Re\left\{\int_0^\infty dt e^{-i\omega t}\langle A^\dagger(t)A\rangle\right\}.
```

This can be done directly by using [`timecorrelations.spectrum`](@ref), which internally calculates the correlation function and the spectrum for a given list of frequencies. Alternatively, if you already calculated the correlation function you can calculate the spectrum and the corresponding frequencies with [`timecorrelations.correlation2spectrum`](@ref).

```@example timecorrelations
ωlist = [-2π:0.1:2π;]
spec = timecorrelations.spectrum(ωlist, H, J, A)
ω, spec = timecorrelations.correlation2spectrum(tspan, g)
nothing # hide
```

If the list of frequencies `ωlist` is omitted, the correlation function is calculated up to steady-state and a list of the frequencies corresponding to the time it takes the system to reach steady-state is returned in addition to the spectrum.

## [Functions](@id timecorrelations: Functions)

* [`timecorrelations.correlation`](@ref)
* [`timecorrelations.spectrum`](@ref)
* [`timecorrelations.correlation2spectrum`](@ref)

## [Examples](@id timecorrelations: Examples)

* [Two-time correlation function and spectrum of a Fock state](@ref)
