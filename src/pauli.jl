module pauli

export PauliBasis, PauliTransferMatrix, DensePauliTransferMatrix,
        ChiMatrix, DenseChiMatrix

import Base: ==
import Base: *

using ..bases, ..spin, ..superoperators
using ..operators: identityoperator, AbstractOperator
using ..superoperators: SuperOperator
using ..operators_dense: DenseOperator
using ..spin: sigmax, sigmay, sigmaz
using SparseArrays: sparse
using LinearAlgebra: tr

"""
    PauliBasis(num_qubits::Int)

Basis for an N-qubit space where `num_qubits` specifies the number of qubits.
The dimension of the basis is 2²ᴺ.
"""
mutable struct PauliBasis{B<:Tuple{Vararg{Basis}}} <: Basis
    shape::Vector{Int}
    bases::B
    function PauliBasis(num_qubits::Int)
        shape = [2 for _ in 1:num_qubits]
        bases = Tuple(SpinBasis(1//2) for _ in 1:num_qubits)
        return new{typeof(bases)}(shape, bases)
    end
end
==(pb1::PauliBasis, pb2::PauliBasis) = length(pb1.bases) == length(pb2.bases)

"""
    Base class for Pauli transfer matrix classes.
"""
abstract type PauliTransferMatrix{B1<:Tuple{PauliBasis, PauliBasis}, B2<:Tuple{PauliBasis, PauliBasis}} end


"""
    DensePauliTransferMatrix(B1, B2, data)

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

PauliTransferMatrix(ptm::DensePauliTransferMatrix{B, B, Matrix{Float64}}) where B <: Tuple{PauliBasis, PauliBasis} = ptm

function *(ptm0::DensePauliTransferMatrix{B, B, Matrix{Float64}},
           ptm1::DensePauliTransferMatrix{B, B, Matrix{Float64}}) where B <: Tuple{PauliBasis, PauliBasis}
    return DensePauliTransferMatrix(ptm0.basis_l, ptm1.basis_r, ptm0.data*ptm1.data)
end

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
                              T<:Matrix{ComplexF64}} <: PauliTransferMatrix{B1, B2}
    basis_l::B1
    basis_r::B2
    data::T
    function DenseChiMatrix(basis_l::BL, basis_r::BR, data::T) where {BL<:Tuple{PauliBasis, PauliBasis},
                                                                                BR<:Tuple{PauliBasis, PauliBasis},
                                                                                T<:Matrix{ComplexF64}}
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new{BL, BR, T}(basis_l, basis_r, data)
    end
end

ChiMatrix(chi_matrix::DenseChiMatrix{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis} = chi_matrix

"""
A dictionary that represents the Pauli algebra - for a pair of Pauli operators
σᵢσⱼ information about their product is given under the key "ij". The first
element of the dictionary value is the Pauli operator, and the second is the
scalar multiplier. For example, σ₀σ₁ = σ₁, and `"01" => ("1", 1)`.
"""
const pauli_multiplication_dict = Dict(
  "00" => ("0", 1.0+0.0im),
  "23" => ("1", 0.0+1.0im),
  "30" => ("3", 1.0+0.0im),
  "22" => ("0", 1.0+0.0im),
  "21" => ("3", -0.0-1.0im),
  "10" => ("1", 1.0+0.0im),
  "31" => ("2", 0.0+1.0im),
  "20" => ("2", 1.0+0.0im),
  "01" => ("1", 1.0+0.0im),
  "33" => ("0", 1.0+0.0im),
  "13" => ("2", -0.0-1.0im),
  "32" => ("1", -0.0-1.0im),
  "11" => ("0", 1.0+0.0im),
  "03" => ("3", 1.0+0.0im),
  "12" => ("3", 0.0+1.0im),
  "02" => ("2", 1.0+0.0im),
)

"""
    multiply_pauli_matirices(i4::String, j4::String)

A function to algebraically determine result of multiplying two
(N-qubit) Pauli matrices. Each Pauli matrix is represented by a string
in base 4. For example, σ₃⊗σ₀⊗σ₂ would be "302". The product of any pair of
Pauli matrices will itself be a Pauli matrix multiplied by any of the 1/4 roots
of 1.
"""
cache_multiply_pauli_matrices() = begin
    local pauli_multiplication_cache = Dict()
    function _multiply_pauli_matirices(i4::String, j4::String)
        if (i4, j4) ∉ keys(pauli_multiplication_cache)
            pauli_multiplication_cache[(i4, j4)] = mapreduce(x -> pauli_multiplication_dict[prod(x)],
                                                             (x,y) -> (x[1] * y[1], x[2] * y[2]),
                                                             zip(i4, j4))
        end
        return pauli_multiplication_cache[(i4, j4)]
    end
end
multiply_pauli_matirices = cache_multiply_pauli_matrices()

function *(chi_matrix0::DenseChiMatrix{B, B, Matrix{ComplexF64}},
           chi_matrix1::DenseChiMatrix{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis}

    num_qubits = length(chi_matrix0.basis_l[1].shape)
    sop_dim = 2 ^ prod(chi_matrix0.basis_l[1].shape)
    ret = zeros(ComplexF64, (sop_dim, sop_dim))

    for ijkl in Iterators.product(0:(sop_dim-1),
                                  0:(sop_dim-1),
                                  0:(sop_dim-1),
                                  0:(sop_dim-1))
        i, j, k, l = ijkl
        if (chi_matrix0.data[i+1, j+1] != 0.0) & (chi_matrix1.data[k+1, l+1] != 0.0)
            i4, j4, k4, l4 = map(x -> string(x, base=4, pad=2), ijkl)

            pauli_product_ik = multiply_pauli_matirices(i4, k4)
            pauli_product_lj = multiply_pauli_matirices(l4, j4)

            ret[parse(Int, pauli_product_ik[1], base=4)+1,
                parse(Int, pauli_product_lj[1], base=4)+1] += (pauli_product_ik[2] * pauli_product_lj[2] * chi_matrix0.data[i+1, j+1] * chi_matrix1.data[k+1, l+1])
        end
    end
    return DenseChiMatrix(chi_matrix0.basis_l, chi_matrix0.basis_r, ret / 2^num_qubits)
end


# TODO MAKE A GENERATOR FUNCTION
"""
    pauli_operators(num_qubits::Int)

Generate a list of N-qubit Pauli operators.
"""
function pauli_operators(num_qubits::Int)
    pauli_funcs = (identityoperator, sigmax, sigmay, sigmaz)
    po = []
    for paulis in Iterators.product((pauli_funcs for _ in 1:num_qubits)...)
        basis_vector = reduce(⊗, f(SpinBasis(1//2)) for f in paulis)
        push!(po, basis_vector)
    end
    return po
end

"""
    pauli_basis_vectors(num_qubits::Int)

Generate a matrix of basis vectors in the Pauli representation given a number
of qubits.
"""
function pauli_basis_vectors(num_qubits::Int)
    po = pauli_operators(num_qubits)
    sop_dim = 4 ^ num_qubits
    return mapreduce(x -> sparse(reshape(x.data, sop_dim)), (x, y) -> [x y], po)
end

"""
    PauliTransferMatrix(sop::DenseSuperOperator)

Convert a superoperator to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(sop::DenseSuperOperator{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(sop.basis_l[1].bases)
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 4 ^ num_qubits
    data = Matrix{Float64}(undef, (sop_dim, sop_dim))
    data .= real.(pbv' * sop.data * pbv / √sop_dim)
    return DensePauliTransferMatrix(sop.basis_l, sop.basis_r, data)
end

SuperOperator(unitary::DenseOperator{B, B, Matrix{ComplexF64}}) where B <: PauliBasis = spre(unitary) * spost(unitary')
SuperOperator(sop::DenseSuperOperator{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis} = sop

"""
    SuperOperator(ptm::DensePauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a superoperator.
"""
function SuperOperator(ptm::DensePauliTransferMatrix{B, B, Matrix{Float64}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(ptm.basis_l[1].bases)
    pbv = pauli_basis_vectors(num_qubits)
    sop_dim = 4 ^ num_qubits
    data = Matrix{ComplexF64}(undef, (sop_dim, sop_dim))
    data .= pbv * ptm.data * pbv' / √sop_dim
    return DenseSuperOperator(ptm.basis_l, ptm.basis_r, data)
end

"""
    PauliTransferMatrix(unitary::DenseOperator)

Convert an operator, presumably a unitary operator, to its representation as a
Pauli transfer matrix.
"""
PauliTransferMatrix(unitary::DenseOperator{B, B, Matrix{ComplexF64}}) where B <: PauliBasis = PauliTransferMatrix(SuperOperator(unitary))

"""
    ChiMatrix(unitary::DenseOperator)

Convert an operator, presumably a unitary operator, to its representation as a χ matrix.
"""
function ChiMatrix(unitary::DenseOperator{B, B, Matrix{ComplexF64}}) where B <: PauliBasis
    num_qubits = length(unitary.basis_l.bases)
    pbv = pauli_basis_vectors(num_qubits)
    aj = pbv' * reshape(unitary.data, 4 ^ num_qubits)
    return DenseChiMatrix((unitary.basis_l, unitary.basis_l), (unitary.basis_r, unitary.basis_r), aj * aj' / (2 ^ num_qubits))
end

"""
    ChiMatrix(sop::DenseSuperOperator)

Convert a superoperator to its representation as a Chi matrix.
"""
function ChiMatrix(sop::DenseSuperOperator{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(sop.basis_l)
    sop_dim = 4 ^ num_qubits
    po = pauli_operators(num_qubits)
    data = Matrix{ComplexF64}(undef, (sop_dim, sop_dim))
    for (idx, jdx) in Iterators.product(1:sop_dim, 1:sop_dim)
        data[idx, jdx] = tr((spre(po[idx]) * spost(po[jdx])).data' * sop.data) / √sop_dim
    end
    return DenseChiMatrix(sop.basis_l, sop.basis_r, data)
end

"""
    PauliTransferMatrix(chi_matrix::DenseChiMatrix)

Convert a χ matrix to its representation as a Pauli transfer matrix.
"""
function PauliTransferMatrix(chi_matrix::DenseChiMatrix{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis}
    num_qubits = length(chi_matrix.basis_l)
    sop_dim = 4 ^ num_qubits
    po = pauli_operators(num_qubits)
    data = Matrix{Float64}(undef, (sop_dim, sop_dim))
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
function SuperOperator(chi_matrix::DenseChiMatrix{B, B, Matrix{ComplexF64}}) where B <: Tuple{PauliBasis, PauliBasis}
    return SuperOperator(PauliTransferMatrix(chi_matrix))
end

"""
    ChiMatrix(ptm::DensePauliTransferMatrix)

Convert a Pauli transfer matrix to its representation as a χ matrix.
"""
function ChiMatrix(ptm::DensePauliTransferMatrix{B, B, Matrix{Float64}}) where B <: Tuple{PauliBasis, PauliBasis}
    return ChiMatrix(SuperOperator(ptm))
end

"""Equality for all varieties of superoperators."""
==(sop1::Union{DensePauliTransferMatrix, DenseSuperOperator, DenseChiMatrix},
   sop2::Union{DensePauliTransferMatrix, DenseSuperOperator, DenseChiMatrix}) = ((typeof(sop1) == typeof(sop2)) &
                                                                                 (sop1.basis_l == sop2.basis_l) &
                                                                                 (sop1.basis_r == sop2.basis_r) &
                                                                                  isapprox(sop1.data, sop2.data))

end # end module
