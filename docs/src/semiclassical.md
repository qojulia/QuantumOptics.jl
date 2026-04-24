# Semi-classical systems

The framework offers an implementation of semi-classical systems and their dynamics. By semi-classical systems we mean systems that are described by a quantum state (a ket or density matrix) and additionally a set of classical variables. Such a state can be implemented using [`semiclassical.State`](@ref).

For example, under certain conditions the motion of a two-level atom can be described classically, while its internal degrees of freedom are still of quantum nature. Say the atom can move in one dimension, e.g. ``x``. The atom's state would then be fully described by a ket ``|\psi\rangle`` that gives the internal state of the atom (meaning whether it is excited or not) and the classical variables for the motion ``x`` and ``p``, where ``x`` is the position in space and ``p`` is the momentum.

Using the framework's implementation, we can define this state by:

```@example semiclassical1
using QuantumOptics # hide
b_atom = SpinBasis(1//2)

ψ = (spinup(b_atom) + spindown(b_atom))/sqrt(2)
ρ = dm(ψ)

x = 0.2
p = 7.5

u = ComplexF64[x, p] # Complex vector containing classical variables

ψ_sc = semiclassical.State(ψ, u)
ρ_sc = semiclassical.State(ρ, u)
nothing # hide
```
Note, that the vector of classical variables **must be complex**.

Furthermore, we can calculate the dynamic, semi-classical time evolution. This can be very practical when, for example, the system Hamiltonian depends on classical variables which themselves have to be calculated dynamically. Much like the time-dependent time evolution solvers [`timeevolution.schroedinger_dynamic`](@ref) and [`timeevolution.master_dynamic`](@ref), the solver requires functional based parameter input. However, instead of just allowing variables to depend on time, these functions can be much more general: the Hamiltonian and jump operators can depend on variables that require simultaneous solution of their own dynamical equations which can in turn depend on the system density matrix. The time evolution can then be obtained by using [`semiclassical.schroedinger_dynamic`](@ref) or [`semiclassical.master_dynamic`](@ref).

The functions, must have the form

```@example semiclassical1
H = 0*one(b_atom) # hide
J, Jdagger = [], [] # hide
# fquantum for a Schrödinger time evolution
function fquantum_schroedinger(t, ψ, u)
  # update H (Hamiltonian) according to dependencies on
  # classical variables u and quantum state ψ
  return H
end

# fquantum for a Master time evolution
function fquantum_master(t, ψ, u)
  # update H (Hamiltonian), J and Jdagger (jump operators)
  # according to dependencies on classical variables
  # u and quantum state ψ
  return H, J, Jdagger
end

# fclassical
function fclassical!(du, u, ψ, t)
  # update du (vector of derivatives of classical variables)
  # according to dependencies on classical variables
  # u and quantum state ψ
  # -- no return statement!
end
nothing # hide
```

Note, that the way these functions are defined has a great impact on the performance of the calculation since they are evaluated at every time step. To save time try to optimize them, e.g. by calculating required operator products outside of the functions.

We can then calculate the time evolution by passing the functions to the solver like so

```@example semiclassical1
tspan = [0:0.1:10;] # hide
tout, ψt = semiclassical.schroedinger_dynamic(tspan, ψ_sc, fquantum_schroedinger, fclassical!)
tout, ρt = semiclassical.master_dynamic(tspan, ψ_sc, fquantum_master, fclassical!)
nothing # hide
```

Computing Monte-Carlo wave function trajectories of a semiclassical system slightly differs in syntax. Namely, when using [`semiclassical.mcwf_dynamic`](@ref) one can also apply "jumps" to the classical part of the system. This is for example the case when considering the recoil an atom experiences when it spontaneously emits a photon. The semiclassical jump is implemented via an additional function, i.e. using the previously defined functions one needs to define an additional one like so:

```@example semiclassical1
function fjump_classical!(u,psi,i,t)
  # update u according to the jump occurring with J[i]
  # -- no return statement!
end
tout, ψt = semiclassical.mcwf_dynamic(tspan, ψ_sc, fquantum_master, fclassical!, fjump_classical!)
nothing # hide
```

Note, that the function `fjump_classical!` takes four arguments: the classical and the quantum parts of the state `u` and `psi`, respectively, the index of which jump occurs `i` and the time `t`. The index `i` is an integer corresponding to the index of the list of jump operators. This can be used when there are multiple possible jumps which act on `u` in a different way.

## [Functions](@id semiclassical: Functions)


* [`semiclassical.State`](@ref)
* [`semiclassical.schroedinger_dynamic`](@ref)
* [`semiclassical.master_dynamic`](@ref)
* [`semiclassical.mcwf_dynamic`](@ref)

## [Examples](@id semiclassical: Examples)

* [Cavity cooling of a two-level atom](@ref)
