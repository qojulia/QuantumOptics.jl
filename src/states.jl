module states

export StateVector, Bra, Ket, length, basis, dagger, tensor,
    norm, normalize, normalize!, permutesystems, basisstate

import Base: ==, +, -, *, /, length, copy, norm, normalize, normalize!
import ..bases: basis, tensor, permutesystems, check_multiplicable, samebases

using Compat
using ..bases


"""
Abstract base class for [`Bra`](@ref) and [`Ket`](@ref) states.

The state vector class stores the coefficients of an abstract state
in respect to a certain basis. These coefficients are stored in the
`data` field and the basis is defined in the `basis`
field.
"""
abstract type StateVector end

"""
    Bra(b::Basis[, data])

Bra state defined by coefficients in respect to the basis.
"""
mutable struct Bra <: StateVector
    basis::Basis
    data::Vector{ComplexF64}
    function Bra(b::Basis, data)
        if length(b) != length(data)
            throw(DimensionMismatch())
        end
        new(b, data)
    end
end

"""
    Ket(b::Basis[, data])

Ket state defined by coefficients in respect to the given basis.
"""
mutable struct Ket <: StateVector
    basis::Basis
    data::Vector{ComplexF64}
    function Ket(b::Basis, data)
        if length(b) != length(data)
            throw(DimensionMismatch())
        end
        new(b, data)
    end
end

Bra(b::Basis) = Bra(b, zeros(ComplexF64, length(b)))
Ket(b::Basis) = Ket(b, zeros(ComplexF64, length(b)))

copy(a::T) where {T<:StateVector} = T(a.basis, copy(a.data))
length(a::StateVector) = length(a.basis)::Int
basis(a::StateVector) = a.basis

==(x::T, y::T) where {T<:StateVector} = samebases(x, y) && x.data==y.data

# Arithmetic operations
+(a::T, b::T) where {T<:StateVector} = (check_samebases(a, b); T(a.basis, a.data+b.data))

-(a::T) where {T<:StateVector} = T(a.basis, -a.data)
-(a::T, b::T) where {T<:StateVector} = (check_samebases(a, b); T(a.basis, a.data-b.data))

*(a::Bra, b::Ket) = (check_multiplicable(a, b); sum(a.data.*b.data))
*(a::Number, b::T) where {T<:StateVector} = T(b.basis, a*b.data)
*(a::T, b::Number) where {T<:StateVector} = T(a.basis, b*a.data)

/(a::T, b::Number) where {T<:StateVector} = T(a.basis, a.data/b)


"""
    dagger(x)

Hermitian conjugate.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))

"""
    tensor(x::Ket, y::Ket, z::Ket...)

Tensor product ``|x⟩⊗|y⟩⊗|z⟩⊗…`` of the given states.
"""
tensor(a::T, b::T) where {T<:StateVector} = T(tensor(a.basis, b.basis), kron(b.data, a.data))
tensor(state::StateVector) = state
tensor(states::T...) where {T<:StateVector} = reduce(tensor, states)

# Normalization functions
"""
    norm(x::StateVector)

Norm of the given bra or ket state.
"""
norm(x::StateVector) = norm(x.data)
"""
    normalize(x::StateVector)

Return the normalized state so that `norm(x)` is one.
"""
normalize(x::StateVector) = x/norm(x)
"""
    normalize!(x::StateVector)

In-place normalization of the given bra or ket so that `norm(x)` is one.
"""
normalize!(x::StateVector) = scale!(x.data, 1./norm(x))

function permutesystems(state::T, perm::Vector{Int}) where T<:StateVector
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, state.basis.shape...)
    data = permutedims(data, perm)
    data = reshape(data, length(data))
    T(permutesystems(state.basis, perm), data)
end

# Creation of basis states.
"""
    basisstate(b, index)

Basis vector specified by `index` as ket state.

For a composite system `index` can be a vector which then creates a tensor
product state ``|i_1⟩⊗|i_2⟩⊗…⊗|i_n⟩`` of the corresponding basis states.
"""
function basisstate(b::Basis, indices::Vector{Int})
    @assert length(b.shape) == length(indices)
    x = zeros(ComplexF64, length(b))
    x[sub2ind(tuple(b.shape...), indices...)] = Complex(1.)
    Ket(b, x)
end

function basisstate(b::Basis, index::Int)
    data = zeros(ComplexF64, length(b))
    data[index] = Complex(1.)
    Ket(b, data)
end


# Helper functions to check validity of arguments
function check_multiplicable(a::Bra, b::Ket)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

samebases(a::T, b::T) where {T<:StateVector} = samebases(a.basis, b.basis)::Bool

end # module
