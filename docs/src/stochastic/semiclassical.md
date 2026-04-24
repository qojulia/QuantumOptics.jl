# [Stochastic semi-classical systems](@id stochastic-semiclassical)

In addition to the stochastic Schrödinger and master equations, an implementation for semi-classical systems that are subject to noise is also available. In general, the functions [`stochastic.schroedinger_semiclassical`](@ref) and [`stochastic.master_semiclassical`](@ref) are written to handle time-dependent semi-classical problems including stochastic processes of any kind. From the point of view of syntax, they are very similar to the semi-classical implementations (see [Semi-classical systems](@ref))

## Semi-classical stochastic Schrödinger equation

To solve semi-classical problems, the functions `fquantum(t, psi, u)` and `fclassical(t, psi, u, du)` need to be passed to the solver. Here, `psi` is the quantum part of a [`semiclassical.State`](@ref), `u` its classical part and `du` the derivative of the classical part. Now, in order to solve a semi-classical equation that is subject to noise, one (or both) of two optional functions needs to be passed to [`stochastic.schroedinger_semiclassical`](@ref). The corresponding keyword arguments are `fstoch_quantum` and `fstoch_classical` that are of the same form as `fquantum` and `fclassical`, respectively.

```@example stochastic-semiclassical
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
Hs = [H] # hide
tspan = [0,0.1] # hide
dt = 0.1 # hide
ψ0 = semiclassical.State(fockstate(b, 0), [0.0im, 0.0im]) # hide
fquantum_schroedinger(t, psi, u) = H # hide
fclassical_schroedinger!(du, u, psi, t) = du # hide
function fstoch_q_schroedinger(t, psi, u)
    # Calculate time-dependent stuff
    Hs
end

function fstoch_c_schroedinger!(du, u, psi, t)
    # Calculate classical stochastic stuff
    du[1] = -u[2] # some example
    du[2] = u[1]
end

# Quantum noise
stochastic.schroedinger_semiclassical(tspan, ψ0, fquantum_schroedinger, fclassical_schroedinger!;
fstoch_quantum=fstoch_q_schroedinger, dt=dt)

# Classical noise
stochastic.schroedinger_semiclassical(tspan, ψ0, fquantum_schroedinger, fclassical_schroedinger!;
fstoch_classical=fstoch_c_schroedinger!, dt=dt)
nothing # hide
```

Note, that here `ψ0` needs to be a [`semiclassical.State`](@ref). If one of the functions is omitted, then the semi-classical problem is solved where noise is only present in the quantum or classical part, respectively. Once again, it is possible to avoid initial calculation of the function `fquantum_stoch` to obtain the length of `Hs` by defining `noise_processes`. This number has to be equal to the length of `Hs`. If `fstoch_classical` is given then you can set this by passing the `noise_prototype_classical` keyword (see [below](@ref noise-combinations)). Note, that this argument becomes necessary when working on combinations of quantum and classical noise, or in cases of non-diagonal classical noise.

## Semi-classical stochastic master equation

Implementing a semi-classical stochastic master equation works similarly to above. The output of the functions needs to be altered in order to return the operators needed for the Lindblad and stochastic superoperator, respectively.

```@example stochastic-semiclassical
J = [destroy(b)] # hide
Jdagger = dagger.(J) # hide
C = J # hide
Cdagger = Jdagger # hide
ρ0 = ψ0 # hide
fquantum_master(t, psi, u) = H, J, Jdagger # hide
fclassical_master!(du, u, psi, t) = du # hide
function fstoch_q_master(t, psi, u)
    # Calculate time-dependent stuff
    C, Cdagger
end

function fstoch_c_master!(du, u, psi, t)
    # Calculate classical stochastic stuff
    du[1] = -u[2] # some example
    du[2] = u[1]
end

# Quantum noise
stochastic.master_semiclassical(tspan, ρ0, fquantum_master, fclassical_master!;
fstoch_quantum=fstoch_q_master, dt=dt)

# Classical noise
stochastic.master_semiclassical(tspan, ρ0, fquantum_master, fclassical_master!;
fstoch_classical=fstoch_c_master!, dt=dt)
nothing # hide
```

Note, that the operators returned by `fstoch_q_master` are cast in the form of a stochastic superoperator in the stochastic master equation. Again, if one of the functions is omitted, the semi-classical time evolution is calculated where noise is only present in the part for which the respective function is defined.

If this is to combined with classical noise, some extra options are necessary.

## [Combinations of quantum and (non-diagonal) classical noise](@id noise-combinations)

While the above discussed examples work fine for problems with quantum noise or classical noise, one needs to be careful when working with combinations of the two. For combinations of quantum and classical noise, we need to set the keyword argument `noise_prototype_classical`. This is essentially the same option as they keyword `noise_rate_prototype` in the **StochasticDiffEq** package, which is needed to treat non-diagonal noise. It is important to note, that even if the quantum noise is diagonal (i.e. there is only one noise operator) and also the classical noise is diagonal, the **combined problem** corresponds to **non-diagonal noise**. This means, that the classical increment `du` in the stochastic classical function is a two-dimensional array. The `noise_prototype_classical` carries the information for the shape of this array.

For example, consider a stochastic Schrödinger equation with a single noise term in the Hamiltonian and diagonal classical noise. This can be implemented by

```@example stochastic-semiclassical
function fstoch_q_diagonal(t, psi, u)
    Hs # This is a vector containing a single operator
end

function fstoch_c_diagonal!(du, u, psi, t)
    # Same example as before, but du is now an array
    du[1,1] = -u[2]
    du[2,2] = u[1]
end

stochastic.schroedinger_semiclassical(tspan, ψ0, fquantum_schroedinger, fclassical_schroedinger!;
fstoch_quantum=fstoch_q_diagonal, fstoch_classical=fstoch_c_diagonal!,
noise_prototype_classical=zeros(ComplexF64, 2, 2), dt=dt)
nothing # hide
```

Note, how we need to index the diagonal of the array `du` in order to treat to obtain a diagonal classical noise problem.

Non-diagonal classical noise can be treated in the same way, e.g.

```@example stochastic-semiclassical
function fstoch_c_nondiag!(du, u, psi, t)
    # Non-diagonal noise
    du[1,1] = -u[2]
    du[1,2] = 0.1u[1]
    du[2,2] = u[1]
    du[2,1] = -0.1u[2]
    du[2,3] = u[2]
end

stochastic.schroedinger_semiclassical(tspan, ψ0, fquantum_schroedinger, fclassical_schroedinger!; dt=dt,
fstoch_classical=fstoch_c_nondiag!, noise_prototype_classical=zeros(ComplexF64, 2, 3))
nothing # hide
```

Note, that the above can be combined with the quantum noise function from before (without any further changes) by passing the corresponding function. The entire discussion above can be used in the same fashion for stochastic master equations.

For details on non-diagonal noise, please refer to the **DifferentialEquations** [documentation](http://docs.juliadiffeq.org/stable/).


## [Functions](@id stochastic-semiclassical: Functions)

* [`stochastic.schroedinger_semiclassical`](@ref)
* [`stochastic.master_semiclassical`](@ref)
