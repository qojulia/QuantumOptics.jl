module nlevel

import Base.==

using ..bases, ..states, ..operators, ..operators_sparse

export NLevelBasis, transition, nlevelstate

"""
Basis for a system consisting of N states.

Arguments
---------

N
    Number of states.
"""
type NLevelBasis <: Basis
    shape::Vector{Int}
    N::Int
    function NLevelBasis(N::Int)
        if N < 1
            throw(DimensionMismatch())
        end
        new([N], N)
    end
end

==(b1::NLevelBasis, b2::NLevelBasis) = b1.N == b2.N


"""
Transition operator :math:`|n><m|`.

Arguments
---------

b
    NLevelBasis
to
    Index of ket state in :math:`|to><from|`
from
    Index of bra state in :math:`|to><from|`
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
State where the system is completely in the n-th level.
"""
function nlevelstate(b::NLevelBasis, n::Int)
    if n < 1 || b.N < n
        throw(BoundsError("n has to be between 1 and b.N"))
    end
    basisstate(b, n)
end

end # module