module states

import Base: ==, +, -, *, /
import ..bases

using Compat
using ..bases

export StateVector, Bra, Ket,
       tensor, dagger, basisstate


"""
Abstract base class for [`Bra`](@ref) and [`Ket`](@ref) states.

The state vector class stores the coefficients of an abstract state
in respect to a certain basis. These coefficients are stored in the
`data` field and the basis is defined in the `basis`
field.
"""
@compat abstract type StateVector end

"""
    Bra(b::Basis[, data])

Bra state defined by coefficients in respect to the basis.
"""
type Bra <: StateVector
    basis::Basis
    data::Vector{Complex128}
    Bra(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

"""
    Ket(b::Basis[, data])

Ket state defined by coefficients in respect to the given basis.
"""
type Ket <: StateVector
    basis::Basis
    data::Vector{Complex128}
    Ket(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

Bra(b::Basis) = Bra(b, zeros(Complex128, length(b)))
Ket(b::Basis) = Ket(b, zeros(Complex128, length(b)))

=={T<:StateVector}(x::T, y::T) = (x.basis == y.basis) && (x.data == y.data)

Base.length(a::StateVector) = length(a.basis)::Int
bases.basis(a::StateVector) = a.basis
Base.copy{T<:StateVector}(a::T) = T(a.basis, copy(a.data))

# Arithmetic operations
*(a::Bra, b::Ket) = (check_multiplicable(a, b); sum(a.data.*b.data))
*{T<:StateVector}(a::Number, b::T) = T(b.basis, complex(a)*b.data)
*{T<:StateVector}(a::T, b::Number) = T(a.basis, complex(b)*a.data)

/{T<:StateVector}(a::T, b::Number) = T(a.basis, a.data/complex(b))

+{T<:StateVector}(a::T, b::T) = (check_samebases(a, b); T(a.basis, a.data+b.data))

-{T<:StateVector}(a::T) = T(a.basis, -a.data)
-{T<:StateVector}(a::T, b::T) = (check_samebases(a, b); T(a.basis, a.data-b.data))

"""
    tensor(x::Ket, y::Ket, z::Ket...)

Tensor product ``|x⟩⊗|y⟩⊗|z⟩⊗…`` of the given states.
"""
bases.tensor{T<:StateVector}(a::T, b::T) = T(tensor(a.basis, b.basis), kron(b.data, a.data))
bases.tensor(state::StateVector) = state
bases.tensor{T<:StateVector}(states::T...) = reduce(tensor, states)

"""
    dagger(x)

Hermitian conjugate.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))


# Normalization functions
"""
    norm(x::StateVector)

Norm of the given bra or ket state.
"""
Base.norm(x::StateVector) = norm(x.data)
"""
    normalize(x::StateVector)

Return the normalized state so that `norm(x)` is one.
"""
Base.normalize(x::StateVector) = x/norm(x)
"""
    normalize!(x::StateVector)

In-place normalization of the given bra or ket so that `norm(x)` is one.
"""
Base.normalize!(x::StateVector) = scale!(x.data, 1./norm(x))


# Creation of basis states.
"""
    basisstate(b, index)

Basis vector specified by `index` as ket state.

For a composite system `index` can be a vector which then creates a tensor
product state ``|i_1⟩⊗|i_2⟩⊗…⊗|i_n⟩`` of the corresponding basis states.
"""
function basisstate(b::Basis, indices::Vector{Int})
    @assert length(b.shape) == length(indices)
    x = zeros(Complex128, length(b))
    x[sub2ind(tuple(b.shape...), indices...)] = Complex(1.)
    Ket(b, x)
end

function basisstate(b::Basis, index::Int)
    data = zeros(length(b))
    data[index] = Complex(1.)
    Ket(b, data)
end

function bases.permutesystems{T<:StateVector}(state::T, perm::Vector{Int})
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    T(permutesystems(state.basis, perm), data)
end


# Helper functions to check validity of arguments
function bases.check_multiplicable(a::Bra, b::Ket)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

bases.samebases{T<:StateVector}(a::T, b::T) = samebases(a.basis, b.basis)::Bool

end # module
