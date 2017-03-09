module states

import Base: ==, +, -, *, /
import ..bases

using ..bases

export StateVector, Bra, Ket,
       tensor, dagger, basisstate


"""
Abstract base class for Bra and Ket.

The state vector class stores the coefficients of an abstract state
in respect to a certain basis. These coefficients are stored in the
``data`` attribute and the basis is defined in the ``basis``
attribute.
"""
abstract StateVector

type Bra <: StateVector
    basis::Basis
    data::Vector{Complex128}
    Bra(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

type Ket <: StateVector
    basis::Basis
    data::Vector{Complex128}
    Ket(b::Basis, data) = length(b) == length(data) ? new(b, data) : throw(DimensionMismatch())
end

Bra(b::Basis) = Bra(b, zeros(Complex128, length(b)))
Ket(b::Basis) = Ket(b, zeros(Complex128, length(b)))

=={T<:StateVector}(x::T, y::T) = (x.basis == y.basis) && (x.data == y.data)

# Arithmetic operations
*(a::Bra, b::Ket) = (check_multiplicable(a.basis, b.basis); sum(a.data.*b.data))
*{T<:StateVector}(a::Number, b::T) = T(b.basis, complex(a)*b.data)
*{T<:StateVector}(a::T, b::Number) = T(a.basis, complex(b)*a.data)

/{T<:StateVector}(a::T, b::Number) = T(a.basis, a.data/complex(b))

+{T<:StateVector}(a::T, b::T) = (check_samebases(a, b); T(a.basis, a.data+b.data))

-{T<:StateVector}(a::T) = T(a.basis, -a.data)
-{T<:StateVector}(a::T, b::T) = (check_samebases(a, b); T(a.basis, a.data-b.data))

"""
Tensor product of given bras or kets.
"""
bases.tensor{T<:StateVector}(a::T, b::T) = T(tensor(a.basis, b.basis), kron(a.data, b.data))


"""
Hermitian conjugate of the given state vector.
"""
dagger(x::Bra) = Ket(x.basis, conj(x.data))
dagger(x::Ket) = Bra(x.basis, conj(x.data))


# Normalization functions
"""
Norm of the given state vector.
"""
Base.norm(x::StateVector) = norm(x.data)

"""
Normalized copy of the given state vector.
"""
Base.normalize(x::StateVector) = x/norm(x)

"""
Normalize the given state vector.
"""
Base.normalize!(x::StateVector) = scale!(x.data, 1./norm(x))


# Creation of basis states.
"""
Ket state where the entry specified by the indices is 1 and all others are zero.
"""
function basisstate(b::Basis, indices::Vector{Int})
    @assert length(b.shape) == length(indices)
    x = zeros(Complex128, reverse(b.shape)...)
    x[reverse(indices)...] = Complex(1.)
    Ket(b, reshape(x, length(b)))
end

"""
Ket state where the i-th entry is 1 and all others are zero.
"""
function basisstate(b::Basis, index::Int)
    data = zeros(length(b))
    data[index] = Complex(1.)
    Ket(b, data)
end

"""
Change the ordering of the subsystems of the given state.

Arguments
---------
state
    A state represented in a composite basis.
perm
    Vector defining the new ordering of the subsystems.
"""
function bases.permutesystems{T<:StateVector}(state::T, perm::Vector{Int})
    @assert length(state.basis.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(state.data, reverse(state.basis.shape)...)
    dataperm = length(perm) - reverse(perm) + 1
    data = permutedims(data, dataperm)
    data = reshape(data, length(data))
    T(permutesystems(state.basis, perm), data)
end


# Helper functions to check validity of arguments
function bases.check_multiplicable(a::Bra, b::Ket)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

function check_samebases{T<:StateVector}(a::T, b::T)
    if a.basis != b.basis
        throw(IncompatibleBases())
    end
end

end # module
