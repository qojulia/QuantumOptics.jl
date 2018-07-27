module subspace

export SubspaceBasis, projector

import Base.==
import ..operators_dense: projector

using ..bases, ..states, ..operators, ..operators_dense


"""
    SubspaceBasis(basisstates)

A basis describing a subspace embedded a higher dimensional Hilbert space.
"""
mutable struct SubspaceBasis <: Basis
    shape::Vector{Int}
    superbasis::Basis
    basisstates::Vector{Ket}
    basisstates_hash::UInt

    function SubspaceBasis(superbasis::Basis, basisstates::Vector{Ket})
        for state = basisstates
            if state.basis != superbasis
                throw(ArgumentError("The basis of the basisstates has to be the superbasis."))
            end
        end
        basisstates_hash = hash([hash.(x.data) for x=basisstates])
        new(Int[length(basisstates)], superbasis, basisstates, basisstates_hash)
    end
end

SubspaceBasis(basisstates::Vector{Ket}) = SubspaceBasis(basisstates[1].basis, basisstates)

==(b1::SubspaceBasis, b2::SubspaceBasis) = b1.superbasis==b2.superbasis && b1.basisstates_hash==b2.basisstates_hash


proj(u::Ket, v::Ket) = dagger(v)*u/(dagger(u)*u)*u

"""
    orthonormalize(b::SubspaceBasis)

Orthonormalize the basis states of the given [`SubspaceBasis`](@ref)

A modified Gram-Schmidt process is used.
"""
function orthonormalize(b::SubspaceBasis)
    V = b.basisstates
    U = Ket[]
    for (k, v)=enumerate(V)
        u = copy(v)
        for i=1:k-1
            u -= proj(U[i], u)
        end
        normalize!(u)
        push!(U, u)
    end
    return SubspaceBasis(U)
end


"""
    projector(b1, b2)

Projection operator between subspaces and superspaces or between two subspaces.
"""
function projector(b1::SubspaceBasis, b2::SubspaceBasis)
    if b1.superbasis != b2.superbasis
        throw(ArgumentError("Both subspace bases have to have the same superbasis."))
    end
    T1 = projector(b1, b1.superbasis)
    T2 = projector(b2.superbasis, b2)
    return T1*T2
end

function projector(b1::SubspaceBasis, b2::Basis)
    if b1.superbasis != b2
        throw(ArgumentError("Second basis has to be the superbasis of the first one."))
    end
    data = zeros(ComplexF64, length(b1), length(b2))
    for (i, state) = enumerate(b1.basisstates)
        data[i,:] = state.data
    end
    return DenseOperator(b1, b2, data)
end

function projector(b1::Basis, b2::SubspaceBasis)
    if b1 != b2.superbasis
        throw(ArgumentError("First basis has to be the superbasis of the second one."))
    end
    data = zeros(ComplexF64, length(b1), length(b2))
    for (i, state) = enumerate(b2.basisstates)
        data[:,i] = state.data
    end
    return DenseOperator(b1, b2, data)
end


end # module
