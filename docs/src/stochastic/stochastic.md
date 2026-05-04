# Stochastics


In addition to the standard time evolution, **QuantumOptics.jl** also features various possibilities to treat stochastic problems. In general, the usage of the [Stochastics](@ref) module is very similar to the standard time evolution. The main difference is, that additional terms that are proportional to the noise in the equation have to be passed to the corresponding functions.

```@example stochastic-intro
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
Hs = [H] # hide
J = [destroy(b)] # hide
Js = J # hide
T = [0,0.1] # hide
ψ0 = fockstate(b, 0) # hide
tout, ψt = stochastic.schroedinger(T, ψ0, H, Hs; dt=1e-2)
tout, ρt = stochastic.master(T, ψ0, H, J, Js; dt=1e-1)
nothing # hide
```

Note, that we need to set the keyword `dt` here, since the default algorithm is a fixed time step method (see below). Like the [Time-evolution](@ref) module, the stochastic solvers are built around [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) using its stochastic module **StochasticDiffEq**. Many of the options available for stochastic problems treated with [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl) like, for example, the choice of algorithm can be used seamlessly within **QuantumOptics.jl**.


### [Default algorithm and noise](@id stochastic-defaults)

The default algorithm is a basic **Euler-Maruyama** method with fixed step size. This choice has been made, since this algorithm is versatile yet easy to understand. Note, that this means that by default, stochastic problems are solved in the **Ito** sense.

To override the default algorithm, simply set the `alg` keyword argument with one of the solvers you found [here](http://docs.juliadiffeq.org/stable/solvers/sde_solve.html#Full-List-of-Methods-1), e.g.

```@example stochastic-intro
import StochasticDiffEq # hide
tout, ψt = stochastic.schroedinger(T, ψ0, H, Hs; alg=StochasticDiffEq.EulerHeun(), dt=1e-2)
nothing # hide
```

Note, that the switch to the `EulerHeun` method solves the problem in the Stratonovich sense.

The default noise is uncorrelated (white noise). Furthermore, since most equations involving quantum noise feature Hermitian noise operators, the noise is chosen to be real. For example,

```@example stochastic-intro
tout, ψt = stochastic.schroedinger(T, ψ0, H, Hs; noise=StochasticDiffEq.RealWienerProcess(0.0, 0.0), dt=1e-1)
nothing # hide
```

corresponds to the default for a single noise term in the Schrödinger equation. Note, that the default is complex noise for semiclassical stochastic equations, where only classical noise is included (for details see [stochastic semiclassical systems](@ref stochastic-semiclassical))

For details on the available algorithms and further control over the solvers, we refer to the [documentation](http://docs.juliadiffeq.org/stable/) of [**DifferentialEquations.jl**](https://github.com/SciML/DifferentialEquations.jl).
