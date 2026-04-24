# Schrödinger time evolution

The Schrödinger equation as one of the basic postulates of quantum mechanics describes the dynamics of a quantum state in a closed quantum system. In Dirac notation the Schrödinger equation and its adjoint equation read

```math
i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H |\Psi(t)\rangle
```
```math
- i\hbar\frac{\mathrm{d}}{\mathrm{d} t} \langle \Psi(t)| = \langle\Psi(t)| H
```

Both versions are implemented and are chosen automatically depending on the type of the provided initial state ([`Ket`](@ref) or [`Bra`](@ref)):

```@example schroedinger
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
tspan = [0,0.1] # hide
psi0 = fockstate(b, 0) # hide
timeevolution.schroedinger(tspan, psi0, H)
nothing # hide
```

The Schrödinger equation solver requires the arguments `tspan`, which is a vector containing the times, the initial state `psi0` as [`Ket`](@ref) or [`Bra`](@ref) and the Hamiltonian `H`.

Additionally, one can pass an output function `fout` as keyword argument. This can be convenient if one directly wants to compute a value that depends on the states, e.g. an expectation value, instead of the states themselves. Consider, for example, a time evolution according to a Schrödinger equation where for all times we want to compute the expectation value of the operator `A`. We can do this by:

```@example schroedinger
tout, psi_t = timeevolution.schroedinger(tspan, psi0, H)
A = destroy(b) # hide
exp_val = expect(A, psi_t)
nothing # hide
```

or equivalently:

```@example schroedinger
exp(t, psi) = expect(A, psi)
tout, exp_val = timeevolution.schroedinger(tspan, psi0, H; fout=exp)
nothing # hide
```

Although the method using `fout` might seem more complicated, it can be very useful for large systems to save memory since instead of all the states we only store one complex number per time step. Note, that `fout` must always be defined with the arguments `(t, psi)`. If `fout` is given, all variables are assigned within `fout` and the call to [`timeevolution.schroedinger`](@ref) returns `nothing`.

We can also calculate the time evolution for a Hamiltonian that is time-dependent. In that case, we need to use the function [`timeevolution.schroedinger_dynamic(tspan, psi0, f::Function)`](@ref). As you can see, this function requires the same arguments as [`timeevolution.schroedinger`](@ref), but a function `f` instead of a Hamiltonian. 


## [Functions](@id schroedinger: Functions)

* [`timeevolution.schroedinger`](@ref)
* [`timeevolution.schroedinger_dynamic`](@ref)


## [Examples](@id schroedinger: Examples)

* [Jaynes-Cummings model](@ref)
* [Particle in harmonic trap potential](@ref)
* [Gaussian wave packet running into a potential barrier](@ref)
* [Entanglement of Two Qubits](@ref)
* [N-Particles in a double-well potential](@ref)
