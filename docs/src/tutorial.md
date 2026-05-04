# Tutorial

In this tutorial we will step through the common tasks necessary to simulate a quantum system. At each step links to documentation and examples, that explain the topic in more detail, will be provided.

In order to use the **QuantumOptics.jl** library it has to be loaded into the current workspace.

```@example tutorial
using QuantumOptics
```


## [Bases](@id Tutorial: Bases)

The first step is always to define a suitable basis for the quantum system under consideration. For example a [`FockBasis`](@ref) can be used to describe a field mode inside of a cavity. It takes an Integer as argument which specifies the photon number cutoff.

```@example tutorial
b = FockBasis(20)
nothing # hide
```

Many quantum systems are already implemented:
* [Fock space](@ref)
* [Charge basis](@ref)
* [Spin](@ref)
* [Particle](@ref)
* [N-Level](@ref)
* [Many-body](@ref)


## States and Operators

For these quantum systems, functions that create the typical operators and states are provided. Typically, they take a basis as argument and create the operator or state in respect to this basis.

```@example tutorial
a = destroy(b)
at = create(b)

alpha = 1.5
psi = coherentstate(b, alpha)
nothing # hide
```

They can be combined in all the expected ways.

```@example tutorial
dpsi = -1im*(a + at)*psi
nothing # hide
```

The [`dagger()`](@ref) function can be used to transform a ket state into a bra state which for example can be used to create a density operator

```@example tutorial
rho = psi ⊗ dagger(psi)
# Alternatively, replace the ⊗ symbol (typed with \otimes)
# by tensor(psi, dagger(psi))
nothing # hide
```

or to calculate expectation values.

```@example tutorial
println("α = ", dagger(psi)*a*psi)
```

Alternatively. the [`expect()`](@ref) function can be used.

```@example tutorial
println("α = ", expect(a, psi))
```


## Composite systems

Most interesting quantum systems consist of several different parts, for example two spins coupled to a cavity mode.

```@example tutorial
ω_atom = 2
ω_field = 1

# 2 level atom described as spin
b_spin = SpinBasis(1//2)
sp = sigmap(b_spin)
sm = sigmam(b_spin)

H_atom = ω_atom*sp*sm

# Use a Fock basis with a maximum of 20 photons to model a cavity mode
b_fock = FockBasis(20)
a = destroy(b_fock)
at = create(b_fock)
n = number(b_fock)

H_field = ω_field*n
nothing # hide
```

Combining operators from those two systems can be done with the [`tensor()`](@ref) function or with the equivalent ``\otimes`` operator.

```@example tutorial
Ω = 1
H_int = Ω*(a ⊗ sp + at ⊗ sm)
nothing # hide
```

To extend the single system Hamiltonians ``H_\mathrm{atom}`` and ``H_\mathrm{spin}`` to the composite system Hilbert space, one possibility is to combine them with identity operators from the opposite sub-system.

```@example tutorial
I_field = identityoperator(b_fock)
I_atom = identityoperator(b_spin)

H_atom_ = I_field ⊗ H_atom
H_field_ = I_atom ⊗ H_field
nothing # hide
```

However, especially for larger systems this can become tedious and it's more convenient to use the [`embed()`](@ref) function.

```@example tutorial
b = b_fock ⊗ b_spin # Basis of composite system
H = embed(b, 1, H_field) + embed(b, 2, H_atom) + H_int
nothing # hide
```

Creating composite states works completely the same.

```@example tutorial
ψ0 = fockstate(b_fock, 1) ⊗ spindown(b_spin)
nothing # hide
```

## [Time evolution](@id Tutorial: Time evolution)

Several different types of [Time-evolution](@ref) are implemented in **QuantumOptics.jl**:
* [Schrödinger time evolution](@ref)
* [Master time evolution](@ref)
* [Monte Carlo wave function time evolution](@ref) (also called *quantum trajectories*)

All of them share a very similar interface so that changing from one to another is mostly done by changing the function name:

```julia
timeevolution.schroedinger(tspan, psi0, H)
timeevolution.master(tspan, psi0/rho0, H, J)
timeevolution.mcwf(tspan, psi0, H, J)
```

They all return two vectors `tout` and `states`.

### Schrödinger equation

Let's now analyze the dynamics of the system according to the Schrödinger equation.

```@example tutorial
tspan = [0:0.05:5;]
tout, ψt = timeevolution.schroedinger(tspan, ψ0, H);
nothing # hide
```

The results can be visualized using for example matplotlib.

```@example tutorial
using PyPlot

figure(figsize=[10, 3])

subplot(1, 2, 1)
xlabel("Time")
ylabel(L"$\langle |e\rangle \langle e| \rangle$")
plot(tout, real(expect(2, sp*sm, ψt)))

subplot(1, 2, 2)
xlabel("Time")
ylabel(L"$\langle n \rangle$")
plot(tout, real(expect(1, n, ψt)));

tight_layout() # hide
savefig("tutorial_schroedinger.svg") # hide
nothing # hide
```

![](tutorial_schroedinger.svg)


### Master equation

Add photon loss to the cavity can be achieved by introducing a jump operator `a`. This means the system is now an open quantum system and is described by a master equation.

```@example tutorial
κ = 1.
J = [embed(b, 1, a)]
tout, ρt = timeevolution.master(tspan, ψ0, H, J; rates=[κ]);

figure(figsize=[10, 3])

subplot(1, 2, 1)
xlabel("Time")
ylabel(L"$\langle |e\rangle \langle e| \rangle$")
plot(tout, real(expect(2, sp*sm, ρt)))

subplot(1, 2, 2)
xlabel("Time")
ylabel(L"$\langle n \rangle$")
plot(tout, real(expect(1, n, ρt)))

tight_layout() # hide
savefig("tutorial_master.svg") # hide
nothing # hide
```

![](tutorial_master.svg)


### Monte Carlo wave function

Alternatively, one can use the MCWF method to analyze the time evolution of the system. Physically, it can be interpreted as an experimental setup where every photon leaving the cavity is measured by a photon counter, thereby projecting the system onto the state ``| \psi\rangle \rightarrow a |\psi\rangle``. This leads to a coherent time evolution according to a Schrödinger equation interrupted by these jumps at certain random points in time.

```@example tutorial
figure(figsize=[10, 3])

subplot(1, 2, 1)
xlabel("Time")
ylabel(L"$\langle |e\rangle \langle e| \rangle$")

subplot(1, 2, 2)
xlabel("Time")
ylabel(L"$\langle n \rangle$")

tout, ψt_mcwf = timeevolution.mcwf(tspan, ψ0, H, J; seed=UInt(0),
                                   display_beforeevent=true,
                                   display_afterevent=true);
subplot(1, 2, 1)
plot(tout, real(expect(2, sp*sm, ψt_mcwf)))
subplot(1, 2, 2)
plot(tout, real(expect(1, n, ψt_mcwf)))
tight_layout() # hide
savefig("tutorial_mcwf.svg") # hide
nothing # hide
```

![](tutorial_mcwf.svg)

In the statistical average the MCWF time evolution is equivalent to the time evolution according to the master equation.

```@example tutorial
Ntrajectories = 10

exp_n = zeros(Float64, length(tspan))
exp_e = zeros(Float64, length(tspan))

function fout(t, psi)
    i = findfirst(isequal(t), tspan)
    N = norm(psi)
    exp_e[i] += real(expect(2, sp*sm, normalize(psi)))
    exp_n[i] += real(expect(1, n, normalize(psi)))
end

using Random; Random.seed!(0)
for i=1:Ntrajectories
    timeevolution.mcwf(tspan, ψ0, H, J; fout=fout)
end

figure(figsize=[10, 3])

subplot(1, 2, 1)
plot(tspan, real(expect(2, sp*sm, ρt)))
plot(tspan, exp_e/Ntrajectories)

subplot(1, 2, 2)
plot(tspan, real(expect(1, n, ρt)))
plot(tspan, exp_n/Ntrajectories)

tight_layout() # hide
savefig("tutorial_mcwf_average.svg") # hide
nothing # hide
```

![](tutorial_mcwf_average.svg)
