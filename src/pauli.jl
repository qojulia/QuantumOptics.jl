module pauli

export PauliBasis, PauliTransferMatrix, ChiMatrix

import Base: ==

using ..bases, ..spin, ..superoperators
using ..operators: identityoperator, AbstractOperator
using ..superoperators: SuperOperator
using ..operators_dense: DenseOperator
using ..spin: sigmax, sigmay, sigmaz
using SparseArrays
using LinearAlgebra: tr

"""
    PauliBasis(num_qubits::Int)

Basis for an N-qubit space where `num_qubits` specifices the number of qubits.
The dimension of the basis is 2²ᴺ.
"""
mutable struct PauliBasis{B<:Tuple{Vararg{Basis}}} <: Basis
    shape::Vector{Int}
    bases::B
    function PauliBasis(num_qubits::Int64)
        return new{Tuple{(SpinBasis{1//2} for _ in 1:num_qubits)...}}([2 for _ in 1:num_qubits], Tuple(SpinBasis(1//2) for _ in 1:num_qubits))
    end
end
==(pb1::PauliBasis, pb2::PauliBasis) = length(pb1.bases) == length(pb2.bases)

"""
Base class for Pauli transfer matrix classes.
"""
abstract type PauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis}, B2<:Tuple{PauliBasis, PauliBasis}} end


"""
    DensePauliTransferMatrix(b, b, data)

DensePauliTransferMatrix stored as a dense matrix.
"""
mutable struct DensePauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis},
                                        B2<:Tuple{PauliBasis, PauliBasis},
                                        T<:Matrix{Float64}} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DensePauliTransferMatrix(basis_l::BL, basis_r::BR, data::T) where {BL<:Tuple{PauliBasis, PauliBasis},
                                                                                BR<:Tuple{PauliBasis, PauliBasis},
                                                                                T<:Matrix{Float64}}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

PauliTransferMatrix(ptm::DensePauliTransferMatrix{B, B, Array{Float64, 2}}) where B <: Tuple{PauliBasis, PauliBasis} = ptm

"""
Base class for χ (process) matrix classes.
"""
abstract type ChiMatrix{B1<:Tuple{PauliBasis, PauliBasis}, B2<:Tuple{PauliBasis, PauliBasis}} end

"""
    DenseChiMatrix(b, b, data)

DenseChiMatrix stored as a dense matrix.
"""
mutable struct DenseChiMatrix{B1<:Tuple{PauliBasis, PauliBasis},
                              B2<:Tuple{PauliBasis, PauliBasis},
                              T<:Matrix{Complex{Float64}}} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DenseChiMatrix(basis_l::BL, basis_r::BR, data::T) where {BL<:Tuple{PauliBasis, PauliBasis},
                                                                                BR<:Tuple{PauliBasis, PauliBasis},
                                                                                T<:Matrix{Complex{Float64}}}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

DenseChiMatrix(chi_matrix::DenseChiMatrix{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis} = chi_matrix

"""
    pauli_operators(num_qubits::Int64)

Generate a matrix of basis vectors in the Pauli representation given a number
of qubits.
"""
function pauli_operators(num_qubits::Int64)
    pauli_funcs = (identityoperator, sigmax, sigmay, sigmaz)
    po = []
    for paulis in Iterators.product((pauli_funcs for _ in 1:num_qubits)...)
        basis_vector = reduce(⊗, f(SpinBasis(1//2)) for f in paulis)
        push!(po, basis_vector)
    end
    return po
end

"""
    pauli_basis_vectors(num_qubits::Int64)

Generate a matrix of basis vectors in the Pauli representation given a number
of qubits.
"""
function pauli_basis_vectors(num_qubits::Int64)
    po = pauli_operators(num_qubits)
    sop_dim = length(po)
    return mapreduce(x -> sparse(reshape(x.data, sop_dim)), (x, y) -> [x y], po)
end

"""
    PauliTransferMatrix(sop::DenseSuperOperator)

Convert a superoperator to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(sop::DenseSuperOperator{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(sop.basis_l[1].bases)
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 2 ^ (2 * num_qubits)
    data = Array{Float64}(undef, (sop_dim, sop_dim))
    data .= real.(pbv' * sop.data * pbv / √sop_dim)
    return DensePauliTransferMatrix(sop.basis_l, sop.basis_r, data)
end

SuperOperator(unitary::DenseOperator{B, B, Array{Complex{Float64},2}}) where B <: PauliBasis = spre(unitary) * spost(unitary')
SuperOperator(sop::DenseSuperOperator{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis} = sop

"""
    SuperOperator(ptm::DensePauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a superoperator.
"""
function SuperOperator(ptm::DensePauliTransferMatrix{B, B, Array{Float64, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(ptm.basis_l[1].bases)
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 2 ^ (2 * num_qubits)
    data = Array{Complex{Float64}}(undef, (sop_dim, sop_dim))
    data .= pbv * ptm.data * pbv' / √sop_dim
    return DenseSuperOperator(ptm.basis_l, ptm.basis_r, data)
end

"""
    PauliTransferMatrix(unitary::DenseOperator)

Convert an operator, presumably a unitary operator, to its representation as a
Pauli transfer matrix.
"""
PauliTransferMatrix(unitary::DenseOperator{B, B, Array{Complex{Float64},2}}) where B <: PauliBasis = PauliTransferMatrix(SuperOperator(unitary))

"""
    ChiMatrix(unitary::DenseOperator)

Convert an operator, presumably a unitary operator, to its representation as a χ matrix.
"""
function ChiMatrix(unitary::DenseOperator{B, B, Array{Complex{Float64},2}}) where B <: PauliBasis
    num_qubits = length(unitary.basis_l.bases)
    pbv = pauli_basis_vectors(num_qubits)
    aj = pbv' * reshape(unitary.data, 2 ^ (2 * num_qubits))
    return DenseChiMatrix((unitary.basis_l, unitary.basis_l), (unitary.basis_r, unitary.basis_r), aj * aj' / (2 ^ num_qubits))
end

"""
    ChiMatrix(sop::DenseSuperOperator)

Convert a superoperator to its representation as a Chi matrix.
"""
function ChiMatrix(sop::DenseSuperOperator{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(sop.basis_l)
    sop_dim = 2 ^ (2 * num_qubits)
    po = pauli_operators(num_qubits)
    data = Array{Complex{Float64}, 2}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = tr((spre(po[idx]) * spost(po[jdx])).data' * sop.data) / √sop_dim
    end
    return DenseChiMatrix(sop.basis_l, sop.basis_r, data)
end

"""
    PauliTransferMatrix(chi_matrix::DenseChiMatrix)

Convert a χ matrix to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(chi_matrix::DenseChiMatrix{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(chi_matrix.basis_l)
    sop_dim = 2 ^ (2 * num_qubits)
    po = pauli_operators(num_qubits)
    data = Array{Float64, 2}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = tr(mapreduce(x -> po[idx] * po[x[1]] * po[jdx] * po[x[2]] * chi_matrix.data[x[1], x[2]],
                                      +,
                                      Iterators.product(1:16, 1:16)).data) / sop_dim |> real
    end
    return DensePauliTransferMatrix(chi_matrix.basis_l, chi_matrix.basis_r, data)
end

"""
    SuperOperator(chi_matrix::DenseChiMatrix)

Convert a χ matrix to its representation as a superoperator.
"""
function SuperOperator(chi_matrix::DenseChiMatrix{B, B, Array{Complex{Float64}, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    return SuperOperator(PauliTransferMatrix(chi_matrix))
end

"""
    ChiMatrix(ptm::PauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a χ matrix.
"""
function ChiMatrix(ptm::DensePauliTransferMatrix{B, B, Array{Float64, 2}}) where B <: Tuple{PauliBasis, PauliBasis}
    return ChiMatrix(SuperOperator(ptm))
end

"""
Equality for all varieties of superoperators.
"""
==(sop1::Union{DensePauliTransferMatrix, DenseSuperOperator, DenseChiMatrix},
   sop2::Union{DensePauliTransferMatrix, DenseSuperOperator, DenseChiMatrix}) = ((typeof(sop1) == typeof(sop2)) &
                                                                                 (sop1.basis_l == sop2.basis_l) &
                                                                                 (sop1.basis_r == sop2.basis_r) &
                                                                                  isapprox(sop1.data, sop2.data))

end # end module
