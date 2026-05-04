# Operators

Operators can be defined as linear maps from one Hilbert space to another. However, equivalently to states, operators in **QuantumOptics.jl** are interpreted as coefficients of an abstract operator in respect to one or more generally two, possibly distinct [Bases](@ref). For a certain choice of bases ``\{|u_i\rangle\}_i`` and ``\{|v_j\rangle\}_j`` an abstract operator ``A`` has the coefficients ``A_{ij}`` which are connected by the relation

```math
A =  \sum_{ij} A_{ij} | u_i \rangle \langle v_j |
```

For this reason all operators define a left hand as well as a right hand basis:

```@example operators
using QuantumOptics # hide
mutable struct MyOperator{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    # ...
end
```

For performance reasons there are three different implementations of operators in **QuantumOptics.jl**, all inheriting from the abstract [`AbstractOperator`](@ref) type:

* [Operators](@ref)
* [Lazy operators](@ref)
* [Time-dependent operators](@ref)

They have the same interface and can in most cases be used interchangeably, e.g. they can be combined using arithmetic functions `*, /, +, -`:

```@example operators
b = SpinBasis(1//2)
sx = sigmax(b)
sy = sigmay(b)
sx + sy
sx * sy
nothing # hide
```

Additionally, the following functions are implemented for all types of operators, if possible:

* [`dagger`](@ref)
* [`identityoperator`](@ref)
* [`tr`](@ref)
* [`normalize`](@ref)
* [`normalize!`](@ref)
* [`expect`](@ref)
* [`variance`](@ref)
* [`tensor`](@ref)
* [`permutesystems`](@ref)
* [`embed`](@ref)
* [`ptrace`](@ref)
* [`exp`](@ref)

### [Operator data and tensor products](@id tensor_order)

The data field of an operator (or a ket/bra) built by a tensor product exhibits reverse ordering to the standard Kronecker product, i.e. `tensor(A, B).data = kron(B.data, A.data)`. This is due to the fact that this order respects the column-major order of stored data in the Julia language which is beneficial for performance. One has to keep this in mind when manipulating the data fields. If desired you can change the data output printed to the REPL with the [`QuantumOpticsBase.set_printing`](@ref) function, i.e. by doing `QuantumOpticsBase.set_printing(standard_order=true)`. Note, that this will only change the displayed output while leaving the respective operator data fields the unmodified.


## [Operators](@id operators_concrete)

[`Operator`](@ref) with a data field represented by a dense array is the default type used for density operators. It is implemented as:

```julia
mutable struct Operator{BL<:Basis,BR<:Basis,T} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
end
```

where the data field can be any type that implements Julia's [AbstractArray interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array-1).

The [`DenseOperator`](@ref) function can be used to construct an [`Operator`](@ref) with a dense array data field, or convert other types of operators to such a type.
Similarly, one can use the [`SparseOperator`](@ref) function to construct an [`Operator`](@ref) with a sparse data field, and convert other types of operators. Also, a method for [`sparse(::AbstractOperator)`](@ref) is provided for conversion.


## Lazy operators

Lazy operators allow delayed evaluation of certain operations. This is useful when combining two operators is numerically expensive but separate multiplication with states is relatively cheap. A nice example is the [`transform`](@ref) implemented for particles. It allows using a fast fourier transformation to convert a state from real space to momentum space, applying a diagonal operator and converting it back. Doing this in operator notation is only fast if the the order of operations is ``\mathrm{IFFT}*(D*(\mathrm{FFT}*\psi))``. To create a Hamiltonian that uses this calculation order, lazy evaluation is needed:

```@example operators
xmin = -5
xmax = 5
Npoints = 100
b_position = PositionBasis(xmin, xmax, Npoints)
b_momentum = MomentumBasis(b_position)

p = momentum(b_momentum)
x = position(b_position)

Tpx = transform(b_momentum, b_position);
Txp = dagger(Tpx)

H_kin = LazyProduct(Txp, p^2/2, Tpx)
H_pot = x^2
H = LazySum(H_kin, H_pot)
nothing # hide
```

In this case the Hamiltonian ``H`` is a lazy sum of the kinetic term ``p^2/2`` and the potential term ``x^2`` where the kinetic term is the lazy product mentioned before. In the end this results in a speed up from ``O(N^2)`` to ``O(N \log N)``.

Besides the above [`LazyProduct`](@ref), there is also an implementation for lazy sums and lazy tensor products. While a [`LazySum`](@ref) works very much identical to the [`LazyProduct`](@ref), a [`LazyTensor`](@ref) is slightly different in terms of implementation. As a brief example, consider the case of two spin-1/2 particles:

```@example operators
b0 = SpinBasis(1//2)
b = tensor(b0, b0)

sm0 = sigmam(b0) # Single spin operator

# Build composite space using lazy tensors
sm1 = LazyTensor(b, [1], (sm0,))
sm2 = LazyTensor(b, [2], (sm0,))

H = LazySum(LazyProduct(dagger(sm1), sm1), LazyProduct(dagger(sm2), sm2))
nothing # hide
```

**Note**


A [`LazyTensor`](@ref) can only consist of [`Operator`](@ref) types when it is to be used with a time evolution. Using, for example, [`LazyProduct`](@ref) to build a [`LazyTensor`](@ref) will result in an error. However, in almost all use cases, one can rewrite these constructs such that [`LazyTensor`](@ref) remains at the lowest level.


See also:

* [`LazyTensor`](@ref)
* [`LazySum`](@ref)
* [`LazyProduct`](@ref)
