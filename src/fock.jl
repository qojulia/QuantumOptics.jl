module fock

export FockBasis, number, destroy, create, displace, fockstate, coherentstate

import Base: ==

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse
using SparseArrays


"""
    FockBasis(N)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Note that the dimension of this basis then is N+1.
"""
mutable struct FockBasis <: Basis
    shape::Vector{Int}
    N::Int
    function FockBasis(N::Int)
        if N < 0
            throw(DimensionMismatch())
        end
        new([N+1], N)
    end
end


==(b1::FockBasis, b2::FockBasis) = b1.N==b2.N

"""
    number(b::FockBasis)

Number operator for the specified Fock space.
"""
function number(b::FockBasis)
    diag = complex.(0.:b.N)
    data = spdiagm(0 => diag)
    SparseOperator(b, data)
end

"""
    destroy(b::FockBasis)

Annihilation operator for the specified Fock space.
"""
function destroy(b::FockBasis)
    diag = complex.(sqrt.(1.:b.N))
    data = spdiagm(1 => diag)
    SparseOperator(b, data)
end

"""
    create(b::FockBasis)

Creation operator for the specified Fock space.
"""
function create(b::FockBasis)
    diag = complex.(sqrt.(1.:b.N))
    data = spdiagm(-1 => diag)
    SparseOperator(b, data)
end

"""
    displace(b::FockBasis, alpha)

Displacement operator ``D(α)`` for the specified Fock space.
"""
displace(b::FockBasis, alpha::Number) = exp(full(alpha*create(b) - conj(alpha)*destroy(b)))

"""
    fockstate(b::FockBasis, n)

Fock state ``|n⟩`` for the specified Fock space.
"""
function fockstate(b::FockBasis, n::Int)
    @assert n <= b.N
    basisstate(b, n+1)
end

"""
    coherentstate(b::FockBasis, alpha)

Coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate(b::FockBasis, alpha::Number, result=Ket(b, Vector{ComplexF64}(length(b))))
    alpha = complex(alpha)
    data = result.data
    data[1] = exp(-abs2(alpha)/2)
    @inbounds for n=1:b.N
        data[n+1] = data[n]*alpha/sqrt(n)
    end
    return result
end

end # module
