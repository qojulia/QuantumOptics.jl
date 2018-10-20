module nlevel

export NLevelBasis, transition, nlevelstate

import Base: ==

using ..bases, ..states, ..operators, ..operators_sparse


"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
mutable struct NLevelBasis{S} <: Basis{S}
    shape::Vector{Int}
    N::Int
    function NLevelBasis{S}(N::Int) where S
        if N < 1
            throw(DimensionMismatch())
        end
        bases.check_bases_parameter(S,1)
        new([N], N)
    end
end
NLevelBasis(N::Int) = NLevelBasis{(N,)}(N)
# ==(b1::NLevelBasis, b2::NLevelBasis) = b1.N == b2.N


"""
    transition(b::NLevelBasis, to::Int, from::Int)

Transition operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|``.
"""
function transition(b::NLevelBasis, to::Int, from::Int)
    if to < 1 || b.N < to
        throw(BoundsError("'to' index has to be between 1 and b.N"))
    end
    if from < 1 || b.N < from
        throw(BoundsError("'from' index has to be between 1 and b.N"))
    end
    op = SparseOperator(b)
    op.data[to, from] = 1.
    op
end


"""
    nlevelstate(b::NLevelBasis, n::Int)

State where the system is completely in the n-th level.
"""
function nlevelstate(b::NLevelBasis, n::Int)
    if n < 1 || b.N < n
        throw(BoundsError("n has to be between 1 and b.N"))
    end
    basisstate(b, n)
end

end # module
