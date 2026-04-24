# States

State vectors in **QuantumOptics.jl** are interpreted as coefficients in respect to a certain [`Basis`](@ref). For example the state ``|\psi\rangle`` can be represented in the basis ``\{|u_i\rangle\}_i`` as ``\psi_i``. These quantities are connected by

```math
|\psi\rangle = \sum_i \psi_i |u_i\rangle
```

and the conjugate equation

```math
\langle\psi| = \sum_i \psi_i^* \langle u_i|.
```

The distinction between coefficients in respect to bra or ket states is strictly enforced which guarantees that algebraic mistakes raise an explicit error:

```@example states
using QuantumOptics # hide
basis = FockBasis(2)
x = Ket(basis, [1,1,1]) # Not necessarily normalized
y = Bra(basis, [0,1,0])
# This throws an error:
# x + y
```

Many of the commonly used states are already implemented in the specific quantum systems, e.g. [`spinup`](@ref) for spins or [`coherentstate`](@ref) for systems described by a fock basis. The [`basisstate`](@ref) function is defined for every basis and is used to generate the i-th basis-state of this basis.

All expected arithmetic functions like `*, /, +, -` are implemented:

```@example states
x + x
x - x
2*x
y*x # Inner product
nothing # hide
```

The hermitian conjugate is performed by the [`dagger`](@ref) function which transforms a bra in a ket and vice versa:

```@example states
dagger(x) # Bra(basis, [1,1,1])
nothing # hide
```

Composite states can be created with the [`tensor`](@ref) function or with the equivalent ``\otimes`` operator:

```@example states
tensor(x, x)
x ⊗ x
tensor(x, x, x)
nothing # hide
```
When working with the `.data` fields of composite states, please keep in mind the order of the data (see [`Operators`](@ref tensor_order) for details).

Alternatively, one can use the tensor function to create a density operator by combining a ket with a bra:

```@example states
tensor(x, dagger(x))
nothing # hide
```

The following functions are also available for states:

* [`norm`](@ref)
* [`normalize`](@ref)
* [`normalize!`](@ref)
* [`ptrace`](@ref)
* [`permutesystems`](@ref)
