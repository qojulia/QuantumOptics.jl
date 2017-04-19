module bases

import Base.==

export Basis, GenericBasis, CompositeBasis, basis,
       tensor, ⊗, ptrace, permutesystems,
       IncompatibleBases,
       samebases, multiplicable,
       check_samebases, check_multiplicable


"""
Abstract base class for all specialized bases.

The Basis class is meant to specify a basis of the Hilbert space of the
studied system. Besides basis specific information all subclasses must
implement a shape variable which indicates the dimension of the used
Hilbert space. For a spin-1/2 Hilbert space this would be the
vector `Int[2]`. A system composed of 2 spins would then have a
shape vector `Int[2 2]`.

Composite systems can be defined with help of the [`CompositeBasis`](@ref)
class.
"""
abstract Basis


"""
    GenericBasis(N)

A general purpose basis of dimension N.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The preferred way is to specify special bases for different systems.
"""
type GenericBasis <: Basis
    shape::Vector{Int}
end

GenericBasis(N::Int) = GenericBasis(Int[N])


"""
    CompositeBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a vector and creates the shape vector directly
from the shape vectors of these subbases. Instead of creating a CompositeBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
type CompositeBasis <: Basis
    shape::Vector{Int}
    bases::Vector{Basis}
end
CompositeBasis(bases::Vector{Basis}) = CompositeBasis(Int[prod(b.shape) for b in bases], bases)
CompositeBasis(bases::Basis...) = CompositeBasis(Basis[bases...])

"""
    tensor(x, y, z...)

Tensor product of the given objects. Alternatively, the unicode
symbol ⊗ (\\otimes) can be used.
"""
tensor() = error("Tensor function needs at least one argument.")

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(b1::Basis, b2::Basis) = CompositeBasis(Int[prod(b1.shape); prod(b2.shape)], Basis[b1, b2])
tensor(b1::CompositeBasis, b2::CompositeBasis) = CompositeBasis(Int[b1.shape; b2.shape], Basis[b1.bases; b2.bases])
function tensor(b1::CompositeBasis, b2::Basis)
    N = length(b1.bases)
    shape = Vector{Int}(N+1)
    bases = Vector{Basis}(N+1)
    for i=1:N
        shape[i] = b1.shape[i]
        bases[i] = b1.bases[i]
    end
    shape[end] = prod(b2.shape)
    bases[end] = b2
    CompositeBasis(shape, bases)
end
function tensor(b1::Basis, b2::CompositeBasis)
    N = length(b2.bases)
    shape = Vector{Int}(N+1)
    bases = Vector{Basis}(N+1)
    for i=1:N
        shape[i+1] = b2.shape[i]
        bases[i+1] = b2.bases[i]
    end
    shape[1] = prod(b1.shape)
    bases[1] = b1
    CompositeBasis(shape, bases)
end
tensor(bases::Basis...) = reduce(tensor, bases)
tensor{T}(x::T...) = reduce(tensor, x)
⊗(a,b) = tensor(a,b)

"""
    length(b::Basis)

Total dimension of the Hilbert space.
"""
Base.length(b::Basis) = prod(b.shape)

"""
    equal_shape(a, b)

Check if two shape vectors are the same.
"""
function equal_shape(a::Vector{Int}, b::Vector{Int})
    if a === b
        return true
    end
    if length(a) != length(b)
        return false
    end
    for i=1:length(a)
        if a[i]!=b[i]
            return false
        end
    end
    return true
end


"""
    equal_bases(a, b)

Check if two subbases vectors are identical.
"""
function equal_bases(a::Vector{Basis}, b::Vector{Basis})
    if a===b
        return true
    end
    for i=1:length(a)
        if a[i]!=b[i]
            return false
        end
    end
    return true
end


==(b1::Basis, b2::Basis) = false
==(b1::GenericBasis, b2::GenericBasis) = equal_shape(b1.shape,b2.shape)
==(b1::CompositeBasis, b2::CompositeBasis) = equal_shape(b1.shape,b2.shape) && equal_bases(b1.bases,b2.bases)


"""
Exception that should be raised for an illegal algebraic operation.
"""
type IncompatibleBases <: Exception end

"""
    samebases(a, b)

Test if two objects have the same bases.
"""
samebases(b1::Basis, b2::Basis) = b1==b2

"""
    check_samebases(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects don't have
the same bases.
"""
function check_samebases(b1, b2)
    if !samebases(b1, b2)
        throw(IncompatibleBases())
    end
end


"""
    multiplicable(a, b)

Check if two objects are multiplicable.
"""
multiplicable(b1::Basis, b2::Basis) = b1==b2

function multiplicable(b1::CompositeBasis, b2::CompositeBasis)
    if !equal_shape(b1.shape,b2.shape)
        return false
    end
    for i=1:length(b1.shape)
        if !multiplicable(b1.bases[i], b2.bases[i])
            return false
        end
    end
    return true
end

"""
    check_multiplicable(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects are
not multiplicable.
"""
function check_multiplicable(b1, b2)
    if !multiplicable(b1, b2)
        throw(IncompatibleBases())
    end
end

"""
    basis(a)

Return the basis of an object.

If it's ambiguous, e.g. if an operator has a different left and right basis,
an [`IncompatibleBases`](@ref) error is thrown.
"""
function basis end


"""
    ptrace(a, indices)

Partial trace of the given basis, state or operator.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are traced out.
"""
function ptrace(b::CompositeBasis, indices::Vector{Int})
    J = [i for i in 1:length(b.bases) if i ∉ indices]
    if length(J)==0
        error("Nothing left.")
    elseif length(J)==1
        return b.bases[J[1]]
    else
        return CompositeBasis(b.shape[J], b.bases[J])
    end
end
ptrace(a, index::Int) = ptrace(a, Int[index])


"""
    permutesystems(a, perm)

Change the ordering of the subsystems of the given object.

For a permutation vector [2,1,3] and a given object with basis [b1, b2, b3]
this function results in [b2, b1, b3].
"""
function permutesystems(b::CompositeBasis, perm::Vector{Int})
    @assert length(b.bases) == length(perm)
    @assert isperm(perm)
    CompositeBasis(b.shape[perm], b.bases[perm])
end

end # module
