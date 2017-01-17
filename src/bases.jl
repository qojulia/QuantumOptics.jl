module bases

import Base.==

export Basis, GenericBasis, CompositeBasis,
       tensor, ⊗, dualbasis, ptrace, permutesystems,
       equal_shape, equal_bases, multiplicable,
       IncompatibleBases,
       check_equal, check_multiplicable


"""
Abstract base class for all specialized bases.

The Basis class is meant to specify a basis of the Hilbert space of the
studied system. Besides basis specific information all subclasses must
implement a shape variable which indicates the dimension of the used
Hilbert space. For a spin 1/2 Hilbert space this would for example be the
vector Int[2]. A system composed of 2 spins could for example have a
shape vector Int[2 2].

Composite systems can be easily defined with help of the CompositeBasis class.
"""
abstract Basis


"""
An easy and fast to use general purpose basis.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The cleaner way is to specify special bases for different systems, i.e.
there are FockBasis, MomentumBasis and PositionBasis.
"""
type GenericBasis <: Basis
    shape::Vector{Int}
end

GenericBasis(N::Int) = GenericBasis(Int[N])


"""
Basis for composite Hilbert spaces.

Simply stores the subbases in a vector and creates the shape vector directly
from the shape vectors of these subbases.
"""
type CompositeBasis <: Basis
    shape::Vector{Int}
    bases::Vector{Basis}
    CompositeBasis(bases::Basis...) = new([prod(b.shape) for b=bases], [bases...])
end


"""
Create composite bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
has another CompositeBasis as subbasis.
"""
tensor() = error("Tensor function needs at least one argument.")
tensor(b1::Basis, b2::Basis) = CompositeBasis(b1, b2)
tensor(b1::CompositeBasis, b2::CompositeBasis) = CompositeBasis(b1.bases..., b2.bases...)
tensor(b1::CompositeBasis, b2::Basis) = CompositeBasis(b1.bases..., b2)
tensor(b1::Basis, b2::CompositeBasis) = CompositeBasis(b1, b2.bases...)
tensor(bases::Basis...) = reduce(tensor, bases)
⊗(a,b) = tensor(a,b)

"""
Return the dual basis to the given basis.

If the basis is orthogonal the dual basis and the basis itself are identitcal.
"""
dualbasis(b::Basis) = b

"""
Total dimension of the Hilbert space.
"""
Base.length(b::Basis) = prod(b.shape)

"""
Check if two shape vectors are the same.

This implementation handles special cases in a very fast way so that their
overhead is very low in often performed operations.
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
Check if two subbases vectors are identical.

This implementation handles special cases in a very fast way so that their
overhead is very low in often performed operations.
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
Check if two objects given in the specified bases are multiplicable.

In the case of orthogonal bases this mostly means that the two bases have to be
the same. For nonorthogonal bases this function can be overridden to account
for their behavior.
"""
multiplicable(b1::Basis, b2::Basis) = b1==dualbasis(b2)

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
Exception that should be raised for an illegal algebraic operation.
"""
type IncompatibleBases <: Exception end


"""
Error throwing version of equality checking.

For a boolean version use the equality operator ==
"""
check_equal(b1::Basis, b2::Basis) = (b1==b2 ? true : throw(IncompatibleBases()))

"""
Error throwing version of multiplicativity checking.

For a boolean version use the function multiplicable.
"""
check_multiplicable(b1::Basis, b2::Basis) = (multiplicable(b1, b2) ? true : throw(IncompatibleBases()))


"""
Partial trace of a composite basis.
"""
function ptrace(b::CompositeBasis, indices::Vector{Int})
    reduced_basis = Basis[]
    for (i, subbasis) in enumerate(b.bases)
        if !(i in indices)
            push!(reduced_basis, subbasis)
        end
    end
    if length(reduced_basis)==0
        error("Nothing left.")
    elseif length(reduced_basis)==1
        return reduced_basis[1]
    else
        return CompositeBasis(reduced_basis...)
    end
end


function permutesystems(bases::Vector{Basis}, perm::Vector{Int})
    @assert length(bases) == length(perm)
    @assert issubset(Set(1:length(bases)), Set(perm))
    bases[perm]
end

"""
Change the ordering of the subbases in a CompositeBasis.

For a permutation vector [2,1,3] and a given composite basis [b1, b2, b3]
this function results in [b2,b1,b3].

Arguments
---------

basis
    A composite basis.
perm
    Vector defining the new ordering of the sub-bases.
"""
bases.permutesystems(basis::CompositeBasis, perm::Vector{Int}) = CompositeBasis(permutesystems(basis.bases, perm)...)


end # module
