# Stochastic Schrödinger equation

The stochastic Schrödinger equation has the basic form

```math
i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = \left(H + \sum_i H_i^s \xi_i(t)\right) |\Psi(t)\rangle
```

where $H_i^s$ are the terms in the Hamiltonian that are each proportional to an independent noise term $\xi_i(t)$. Note, that by default the noise is assumed to be real white-noise (uncorrelated in time), i.e. $\langle\xi_i(t)\xi_i(t')\rangle = \delta(t-t')$.

The above equation is straightforward to implement with

```@example stochastic-schroedinger
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
Hs = [H] # hide
tspan = [0,0.1] # hide
ψ0 = fockstate(b, 0) # hide
dt = 0.1 # hide
stochastic.schroedinger(tspan, ψ0, H, Hs; dt=dt)
nothing # hide
```

Here, the first three arguments are the same as for [`timeevolution.schroedinger`](@ref). The additional argument `Hs` is either an [`AbstractOperator`](@ref) or a vector of operators. The first case corresponds to the stochastic Schrödinger equation featuring only a single noise process, while the latter corresponds to a number of noise processes, where the operators $H_i^s$ are the entries in the vector `Hs`.

A time-dependent stochastic Schrödinger equation can also be implemented using [`stochastic.schroedinger_dynamic`](@ref). Instead of the two parts of the Hamiltonian `H` and `Hs`, the dynamic version takes two functions as arguments. One, that returns the deterministic part of the dynamic Hamiltonian, while the other returns a vector with the (time- or state-dependent) operators $H_i^s$ as entries.

```@example stochastic-schroedinger
tspan = [0,0.1] # hide
ψ0 = fockstate(b, 0) # hide
function fdeterm(t, psi)
    # Calculate time-dependent stuff
    # Update deterministic part of Hamiltonian
    H
end

function fstoch(t, psi)
    # Calculate time-dependent stuff
    # Update stochastic part(s) of Hamiltonian
    Hs # This has to be a vector
end

stochastic.schroedinger_dynamic(tspan, ψ0, fdeterm, fstoch; dt=dt)
nothing # hide
```

Note, that the solver requires to know the number of noise processes. This number is calculated automatically from the given function `fstoch` by calculating the output at time `t=0`. If you want to avoid an initial execution of the function, you can make the solver skip the initial calculation of `fstoch`. This is done by passing the number of noise processes, i.e. the length of the vector `Hs` that is returned by `fstoch`, using the optional argument `noise_processes`.

For some problems, it can be useful to renormalize the state vector after every time step taken (this can be used to avoid numerical issues for problems where the norm can become very small/large). To do this here, you can simply set the keyword `normalize_state=true`, e.g.

```@example stochastic-schroedinger
stochastic.schroedinger(tspan, ψ0, H, Hs; dt=dt, normalize_state=true)
nothing # hide
```

To be clear what this renormalization does, it uses a so-called `FunctionCallingCallback` from **DifferentialEquations.jl's** callback library (see the documentation). So doing the following

```@example stochastic-schroedinger
import DiffEqCallbacks # hide
norm_func(u::Vector{ComplexF64}, t::Float64, integrator) = normalize!(u)
ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
                 func_everystep=true, func_start=false)

stochastic.schroedinger(tspan, ψ0, H, Hs; dt=dt, callback=ncb)
nothing # hide
```

is the same as setting `normalize_state=true`. See also **DifferentialEquations.jl's** [callback library](http://docs.juliadiffeq.org/latest/features/callback_library.html#FunctionCallingCallback-1).


## [Detection schemes with the stochastic Schrödinger equation ](@id schroedinger-homodyne)

Currently, **QuantumOptics.jl** features only one pre-defined measurement scheme to be used with the stochastic Schrödinger equation. Namely, with the function [`stochastic.homodyne_carmichael`](@ref), one can obtain the function `fdeterm` and `fstoch` needed to describe homodyne detection as derived by H. J. Carmichael. These can then be used with [`stochastic.schroedinger_dynamic`](@ref). Note, that this version does not preserve the norm.

Consider a system descibed by the Hamiltonian $H_0$. If it is subject to decay with an operator $C$ and the output from this decay channel is subject to homodyne detection, the problem at hand is completely described by the stochastic state-dependent Hamiltonian [1,2]

$H = H_0 - \frac{i}{2}C^\dagger C + iC e^{-i\theta}\left(\xi(t) + \langle C e^{-i\theta} + C^\dagger e^{i\theta}\rangle\right).$

Here, $\theta$ is the phase difference between the signal field (from the decay channel) and the local oscillator. In order to implement this, one needs to split this according to the deterministic and stochastic parts of the Hamiltonian. Additionally, the deterministic part needs to account for the nonlinear term (expectation value). This can be done by, for example,

```@example stochastic-schroedinger
H0 = number(b) # hide
C = destroy(b) # hide
θ = 0.5π
Hs = 1.0im*C*exp(-1.0im*θ)
Y = C*exp(-1.0im*θ) + dagger(C)*exp(1.0im*θ)
CdagC = -0.5im*dagger(C)*C
H_nl(ψ) = expect(Y, normalize(ψ))*Hs + CdagC

fdet_homodyne(t, ψ) = H0 + H_nl(ψ)
fst_homodyne(t, ψ) = [Hs]
stochastic.schroedinger_dynamic(tspan, ψ0, fdet_homodyne, fst_homodyne; dt=dt, normalize_state=true)
nothing # hide
```

The above code would calculate a stochastic trajectory for the measurement of the y-quadrature ($\theta = \pi/2$). Since this scheme is quite commonly used, there is a function which defines the (performance optimized) functions `fdeterm` and `fstoch`. Hence, one can implement the above example in a much more simple way:

```@example stochastic-schroedinger
θ = 0.5π
fdet_cm, fst_cm = stochastic.homodyne_carmichael(H0, C, θ)
stochastic.schroedinger_dynamic(tspan, ψ0, fdet_cm, fst_cm; dt=dt, normalize_state=true)
nothing # hide
```


## [Functions](@id stochastic-schroedinger: Functions)

* [`stochastic.schroedinger`](@ref)
* [`stochastic.schroedinger_dynamic`](@ref)


## [Examples](@id stochastic-schroedinger: Examples)

* [Two-level atom driven by a noisy laser](@ref)

## [References](@id stochastic-schroedinger: References)

[1] Carmichael, Howard. An open systems approach to quantum optics: Lectures presented at the Université Libre de Bruxelles, October 28 to November 4, 1991. Vol. 18. Springer Science & Business Media, 2009.

[2] Wiseman, H. M., & Milburn, G. J. (1993). Quantum theory of field-quadrature measurements. Physical review A, 47(1), 642.
