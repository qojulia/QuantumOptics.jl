module operators_sparse

export diagonaloperator

import Base: ==, *, /, +, -
import ..operators
import SparseArrays: sparse

using ..bases, ..states, ..operators, ..operators_dense, ..sparsematrix
using SparseArrays, LinearAlgebra

const OperatorDataType = Union{Matrix{ComplexF64},SparseMatrixCSC{ComplexF64,Int}}

operators.dense(x::Operator{BL,BR,T}) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} =
    Operator(x.basis_l, x.basis_r, Matrix(x.data))

"""
    sparse(op::AbstractOperator)

Convert an arbitrary operator into a [`SparseOperator`](@ref).
"""
sparse(a::AbstractOperator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(dense(op)) instead."))
sparse(a::Operator{BL,BR,T}) where {BL<:Basis,BR<:Basis,T<:OperatorDataType} = Operator{BL,BR,SparseMatrixCSC{ComplexF64,Int}}(a.basis_l, a.basis_r, sparse(a.data))

function operators.ptrace(op::Operator{BL,BR,T}, indices::Vector{Int}) where {BL<:Basis,BR<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}}
    operators.check_ptrace_arguments(op, indices)
    shape = [op.basis_l.shape; op.basis_r.shape]
    data = sparsematrix.ptrace(op.data, shape, indices)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    SparseOperator(b_l, b_r, data)
end

function operators.expect(op::Operator{BL,BR,T1}, state::Operator{BR,BL,T2}) where {BL<:Basis,BR<:Basis,T1<:SparseMatrixCSC{ComplexF64,Int},T2<:OperatorDataType}
    result = ComplexF64(0.)
    @inbounds for colindex = 1:op.data.n
        for i=op.data.colptr[colindex]:op.data.colptr[colindex+1]-1
            result += op.data.nzval[i]*state.data[colindex, op.data.rowval[i]]
        end
    end
    result
end

function operators.permutesystems(rho::Operator{BL,BR,T}, perm::Vector{Int}) where {BL<:CompositeBasis,BR<:CompositeBasis,T<:SparseMatrixCSC{ComplexF64,Int}}
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert isperm(perm)
    shape = [rho.basis_l.shape; rho.basis_r.shape]
    data = sparsematrix.permutedims(rho.data, shape, [perm; perm .+ length(perm)])
    Operator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

operators.identityoperator(b1::BL, b2::BR) where {BL<:Basis,BR<:Basis} = identityoperator(Operator{BL,BR,SparseMatrixCSC{ComplexF64, Int}}, b1, b2)
operators.identityoperator(b::Basis) = identityoperator(b, b)

"""
    diagonaloperator(b::Basis)

Create a diagonal operator of type [`SparseOperator`](@ref).
"""
function diagonaloperator(b::B, diag::Vector{T}) where {B<:Basis,T<:Number}
  @assert 1 <= length(diag) <= prod(b.shape)
  Operator{B,B,SparseMatrixCSC{ComplexF64,Int}}(b, sparse(Diagonal(convert(Vector{ComplexF64}, diag))))
end

end # module
