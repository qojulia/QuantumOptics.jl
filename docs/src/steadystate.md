# Steady state

**QuantumOptics.jl** implements three different ways to calculate steady states. The first one is to perform a time evolution according to a master equation until an adequate accuracy is reached

* [`steadystate.master`](@ref)

Most of the arguments are identical to the ones used for a standard [Master time evolution](@ref). There are a few differences, however. First of all it is no longer required to give an initial state, but rather optional. If no initial state is given it defaults to a density matrix with a single 1 in the first entry. Note, that while for a cavity this corresponds to a Fock state with zero photons, it might be undesirable to use the default value when working with atoms (spin particles), since this state corresponds to the (highest) excited state.

Additionally, one can explicitly specify the trace distance criterion tolerance with the keyword argument `tol`, which is checked in every time step. The time evolution stops, i.e. steady-state is reached, when the trace distance between two consecutive density matrices ``\rho(t)`` and ``\rho(t+dt)`` is smaller than this value. The default is `tol=1e-3`.

For smaller system sizes finding eigenvectors of super-operators is the preferred method:

* [`steadystate.eigenvector`](@ref)

Finally, for large systems that require a high accuracy, the recommended approach is to use the iterative steady-state solver:

* [`steadystate.iterative`](@ref)

This method iteratively solves the equation ``L*\rho =  0`` using [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl). The default method used is the stabilized bi-conjugate gradient method (`bicgstabl!`). Other methods may be used by specifying the `method!` keyword argument. For details on the available algorithms, please refer to the [documentation of IterativeSolvers.jl](https://iterativesolvers.julialinearalgebra.org/dev/). Additionally, the Liouvillian is implemented as a linear map using [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl) rather than a matrix. This makes the procedure memory-efficient.

A simple example, that can also be solved analytically is to compute the steady-state photon number of a cavity driven with an amplitude ``\eta`` and damping rate ``2\kappa``. The analytical solution is ``n=\eta^2/\kappa^2``, while in the toolbox we may calculate this with:

```@example steadystate
using QuantumOptics # hide
b = FockBasis(10)
κ = 1.0
η = 0.5κ
H = η*(destroy(b) + create(b))
J = [sqrt(2κ)*destroy(b)]
tout, ρ_master = steadystate.master(H, J)
ρ_eig = steadystate.eigenvector(H, J)
ρ_it = steadystate.iterative(H,J)
println("n_master = ", real(expect(number(b), ρ_master[end])))
println("n_eig = ", real(expect(number(b), ρ_eig)))
println("n_it = ", real(expect(number(b), ρ_it)))
nothing # hide
```

Note, that [`steadystate.master`](@ref) returns a list of times and density matrices containing the initial time/state and the time/state when steady state is reached. Due to numerical errors the results are not exactly ``0.25``. Decreasing the value of `tol` below its default `1e-3` improves the accuracy for the time-evolution approach.


## [Functions](@id steadstate: Functions)

* [`steadystate.master`](@ref)
* [`steadystate.eigenvector`](@ref)
* [`steadystate.iterative`](@ref)
* [`steadystate.iterative!`](@ref)

## [Examples](@id steadystate: Examples)

* [Optomechanical Cavity](@ref)
