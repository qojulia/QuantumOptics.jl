# Stochastic Master equation

A stochastic master equation has the general form

```math
\dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
             + \mathcal{L}[\rho] + \mathcal{H}[\rho]
```

where

```math
\mathcal{L}[\rho]=\sum_i \left(
       J_i \rho J_i^\dagger
       - \frac{1}{2} J_i^\dagger J_i \rho
       - \frac{1}{2} \rho J_i^\dagger J_i
   \right)
```

and

```math
\mathcal{H}[\rho] = \sum_n\left[
    \left(C_n\rho + \rho C_n^\dagger\right) - \langle C_n + C_n^\dagger\rangle\rho\right]\xi_n(t)
```

Here, $J_i$ are the Lindblad damping operators, while $C_n$ are operators proportional to the white noise terms $\xi_n(t)$. The last term in $\mathcal{H}[\rho]$ (the expectation value) ensures trace conservation.

In order to describe a measurement, one has to set the operators $C_n$ proportional to collapse operators [1]. The superoperator $\mathcal{H}$ then describes the information gain due to a measurement. For example, to describe unit-efficiency homodyne detection in a single cavity mode, one has to set $C = \sqrt{\kappa}a e^{-i\theta}$, where $\kappa$ is the damping rate, $a$ the photon annihilation operator and $\theta$ is the phase difference between the signal field and the local oscillator.

Note, that (if desired) stochastic terms $H_n^s$ in the Hamiltonian can be included by setting $C_n = -i H_n^s$. If the operators $H_n^s$ are Hermitian, then the $\mathcal{H}[\rho]$ simply becomes a commutator and the expectation value vanishes (since the commutator preserves the trace).

The function that implements this equation is very similar to [`timeevolution.master`](@ref).

```@example stochastic-master
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
J = [destroy(b)] # hide
C = J # hide
tspan = [0,0.1] # hide
dt = 0.1 # hide
ρ0 = fockstate(b, 0) # hide
stochastic.master(tspan, ρ0, H, J, C; dt=dt)
nothing # hide
```

The only additional argument here is `C`, which is a vector containing the operators $C_n$ which constitute the superoperator defined in the stochastic master equation above.

For time-dependent problems, one can make use of the dynamic version. You simply need to define two functions for the deterministic and the stochastic part of the master equation, respectively.

```@example stochastic-master
Jdagger = dagger.(J) # hide
Cdagger = dagger.(C) # hide
function fdeterm(t, rho)
    # Calculate time-dependent stuff
    H, J, Jdagger
end

function fstoch(t, rho)
    # Calculate time-dependent stuff
    C, Cdagger
end
stochastic.master_dynamic(tspan, ρ0, fdeterm, fstoch; dt=dt)
nothing # hide
```

Note, that `C` and `Cdagger` have to be vectors of equal length containing the operators $C_n$.
Optionally, one can include rates in the function output, `fdeterm(t, rho) = H, J, Jdagger, rates` for the deterministic part.

Like in the [`stochastic.schroedinger_dynamic`](@ref) function, if you want to avoid initial calculation of the given functions in order to find the total number of noise processes, you can do so by passing the `noise_processes` keyword argument. Note, that `noise_processes` has to be equal to `length(C)`.

## [Functions](@id stochastic-master: Functions)

* [`stochastic.master`](@ref)
* [`stochastic.master_dynamic`](@ref)

## [Examples](@id stochastic-master: Examples)

* [Quantum Zeno Effect](@ref)

## [References](@id stochastic-master: References)

[1] Jacobs, K. and Steck, D. A. A straighforward introduction to continuous quantum measurements, Contemporary Physics, 47:5, 279-303, (2006). URL: [https://arxiv.org/abs/quant-ph/0611067](https://arxiv.org/abs/quant-ph/0611067)
