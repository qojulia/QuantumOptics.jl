module nlevel

import Base.==

using ..bases, ..states, ..operators, ..operators_sparse

export NLevelBasis, transition


type NLevelBasis <: Basis
    shape::Vector{Int}
    energies::Vector{Float64}
    function NLevelBasis(energies::Vector{Float64})
        new([length(energies)], energies)
    end
end

==(b1::NLevelBasis, b2::NLevelBasis) = b1.energies == b2.energies

function transition(b::NLevelBasis, to::Int, from::Int)
    op = SparseOperator(b)
    op.data[to, from] = 1.
    op
end


end # module