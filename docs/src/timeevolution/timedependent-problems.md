# Time-dependent problems

Each of the time evolution functions offers a variant that can handle arbitrary time and state dependencies in the operators defining the problem. 
These functions accept time-dependent operators in two different forms:
1. as user-provided operator-valued functions of time
2. as subtypes of [`AbstractTimeDependentOperator`](@ref)

The former is completely general, but care must be taken to ensure good performance.
The latter is has fewer performance caveats and is typically more convenient, but
requires that the time-dependence of your problem takes certain supported forms.

## Time-dependent operators

Time-dependent operators are subtypes of [`AbstractTimeDependentOperator`](@ref).
Each such operator has its own internal "clock" set to a time `t` and hence
always has a concrete value.

Operator clocks can be read with [`current_time`](@ref) and set either with
[`set_time!`](@ref) or by calling the operator itself as a function of `t` like
`my_operator(t)`.

Time evolution functions use [`set_time!`](@ref) to update evolution operators
to the current integrator time.

Composing time-dependent operators, say by summing them, is only allowed if their
clocks are set to the same time.

To retrieve a static representation of a time-dependent operator (with the
time dependence stripped away), use [`static_operator`](@ref).

## Time-dependent sums

The [`TimeDependentSum`](@ref) operator is an [`AbstractTimeDependentOperator`](@ref)
representing a sum of constant operators multiplied by time-dependent coefficients.
These are operators of the form 
```math
H(t) = \sum_k c_k(t) h_k,
```
where `t` is time. They can be constructed directly, mixing static and time-dependent
coefficients, for example
* `TimeDependentSum([1.0, cos, (t->10.0 * sin(5*t))], [H_static, H_drive1, H_drive2])`
* `TimeDependentSum(1.0=>H_static, cos=>H_drive1, (t->10.0 * sin(5*t))=>H_drive2)`

or by composition
```julia
H = H_static + TimeDependentSum(cos=>H_drive1) + 10 * TimeDependentSum((t->sin(5*t))=>H_drive2)
```

## Closed system evolution

For closed systems with a time-dependent Hamiltonian, you can use [`timeevolution.schroedinger_dynamic(tspan, psi0, H::Function)`](@ref) to obtain a solution. The function `H(t,psi)` needs to take the current time `t` and state `psi` as input and return a valid Hamiltonian. As a brief example, consider a spin-1/2 particle that is coherently driven by a laser that has an amplitude that varies in time. We can implement this with:

```@example timeevolution_dynamic
using QuantumOptics
basis = SpinBasis(1//2)
ψ₀ = spindown(basis)
const sx = sigmax(basis)
function H_pump(t, psi)
  return sin(t)*sx
end
tspan = [0:0.1:10;]
tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, H_pump)
nothing # hide
```

Alternatively, using [`TimeDependentSum`](@ref):
```@example timeevolution_dynamic_tdop
using QuantumOptics
basis = SpinBasis(1//2)
ψ₀ = spindown(basis)
sx = sigmax(basis)
H_pump_tdop = TimeDependentSum(sin=>sx)
tspan = [0:0.1:10;]
tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, H_pump_tdop)
nothing # hide
```


## Open system evolution

Similarly to the above, we can also use [`timeevolution.master_dynamic(tspan, rho0, f::Function)`](@ref) and [`timeevolution.mcwf_dynamic(tspan, rho0, f::Function)`](@ref), respectively, to simulate open-system dynamics with time-dependent operators. The function still takes the current time and state as input. However, it needs to return the Hamiltonian, as well as the respective jump operators at every time step. To illustrate, let's add spontaneous emission to the example above:

```@example timeevolution_dynamic
const J = [sigmam(basis)]
const Jdagger = [sigmap(basis)]
function f(t,rho)
    H = sin(t)*sx
    return H, J, Jdagger
end
tspan = [0:0.1:10;]
tout, ρₜ = timeevolution.master_dynamic(tspan, ψ₀, f)
nothing # hide
```

Optionally, we can also return four arguments, where the last one specifies the list of (potentially time-dependent) decay rates. The same function profile is required for [`timeevolution.mcwf_dynamic`](@ref).

!!! note

    The state passed to the dynamic function is still used by the integrator. Do not modify the state directly! Also, when using the state in a dynamic function in [`timeevolution.mcwf_dynamic`](@ref), e.g. to compute expectation values, keep in mind that the state is not normalized. Expectation values need to be divided by the norm to ensure correctness.

With [`TimeDependentSum`](@ref):
```@example timeevolution_dynamic_tdop
J = [sigmam(basis)]
H_pump_tdop = TimeDependentSum(sin=>sx)
tspan = [0:0.1:10;]
tout, ρₜ = timeevolution.master_dynamic(tspan, ψ₀, H_pump_tdop, J)
nothing # hide
```

## Comment on performance with the functional approach

The functional approach to time-dependent problems gives you a lot of freedom as you can implement almost arbitrary time and state dependencies. However, it also makes you responsible for performance as the functions passed into the time evolution are time-critical (they are evaluated at every time step the solver takes). For general tips on writing performant Julia code, see the official [Julia manual](https://docs.julialang.org/en/v1/manual/performance-tips/). To summarize, the most relevant points are:

- Precompute everything you can: try not to allocate new operators in the time-dynamic functions. Allocating matrices will greatly impact performance.
- Avoid non-constant globals: if you precompute an operator in global scope, make sure to make it `const` in order for things to be type-stable in the function.
- Try to exploit the specific structure of your Hamiltonian/operators to update them in-place if you can (see also below).

To check for type-instabilities you can use Julia's `@code_warntype`. While trying to optimize a function the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) package is quite useful too.


### Efficient solution for Hamiltonians with time-dependent coefficients

You will commonly encounter a time-dependent Hamiltonian of the form

```math
H(t) = \sum_k c_k(t) h_k,
```

where ``c_k(t)`` are some time-dependent numbers and ``h_k`` are constant operators. Recomputing this sum can be expensive. Fortunately, we can use a [`LazySum`](@ref) which stores the coefficients and operators separately and evaluates it only when the operator is multiplied with a state. This also allows us to update the coefficients in-place at every time step. In fact, we can use this approach to optimize the short example shown above:

```@example timedependent-coefficients
using QuantumOptics
basis = SpinBasis(1//2)
ψ₀ = spindown(basis)
sx = sigmax(basis)
const H = LazySum([0.0],[sx])
function H_pump(t, psi)
  H.factors[1] = sin(t)
  return H
end
tspan = [0:0.1:10;]
tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, H_pump)
nothing # hide
```

Or, for a slightly more involved example, say you would like to apply a series of Gaussian pulses to two qubits:

```@example serial-pulses
using QuantumOptics

# Generic Gaussian pulse
pulse(t,t0,Ω) = @. Ω*exp(-(t-t0)^2)

# Operators
b1 = SpinBasis(1//2)
sx1 = tensor(sigmax(b1), one(b1))
sx2 = tensor(one(b1), sigmax(b1))

# Define coefficients and Hamiltonian
tspan = [0.0:0.1:10.0;]
const coeff_funcs = [t->pulse(t,1,0.5),t->(pulse(t,5,1))]
const H = LazySum([c(tspan[1]) for c∈coeff_funcs],[sx1,sx2])

# Dynamic function
function Ht(t,psi)
    for i=1:length(H.factors)
        H.factors[i] = coeff_funcs[i](t)
    end
    return H
end

psi0 = tensor(spindown(b1), spindown(b1))
tout, psi_t = timeevolution.schroedinger_dynamic(tspan, psi0, Ht)
nothing # hide
```

!!! note

    Using [`TimeDependentSum`](@ref) achieves essentially the same thing as this
    example using [`LazySum`](@ref) and is preferred for these cases.

#### Sampling discrete coefficients

If you have a set of coefficients whose values you only know at certain points in time, you can use an interpolation library such as [Interpolation](https://github.com/JuliaMath/Interpolations.jl) to obtain a continuous approximation of them. Then, the same approach as shown above works.


## [Functions](@id dynamic: Functions)

* [`timeevolution.schroedinger_dynamic`](@ref)
* [`timeevolution.master_dynamic`](@ref)
* [`timeevolution.master_nh_dynamic`](@ref)
* [`timeevolution.mcwf_dynamic`](@ref)
* [`timeevolution.mcwf_nh_dynamic`](@ref)
