module operators_sparse

export SparseOperator, diagonaloperator

import Base: ==, *, /, +, -
import ..operators
import SparseArrays: sparse

using ..bases, ..states, ..operators, ..operators_dense, ..sparsematrix
using SparseArrays, LinearAlgebra


"""
    SparseOperator(b1[, b2, data])

Sparse array implementation of Operator.

The matrix is stored as the julia built-in type `SparseMatrixCSC`
in the `data` field.
"""
mutable struct SparseOperator <: Operator
    basis_l::Basis
    basis_r::Basis
    data::SparseMatrixCSC{ComplexF64, Int}
    function SparseOperator(b1::Basis, b2::Basis, data)
        if length(b1) != size(data, 1) || length(b2) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(b1, b2, data)
    end
end

SparseOperator(b::Basis, data::SparseMatrixCSC{ComplexF64, Int}) = SparseOperator(b, b, data)
SparseOperator(b::Basis, data::Matrix{ComplexF64}) = SparseOperator(b, sparse(data))
SparseOperator(op::DenseOperator) = SparseOperator(op.basis_l, op.basis_r, sparse(op.data))

SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, spzeros(ComplexF64, length(b1), length(b2)))
SparseOperator(b::Basis) = SparseOperator(b, b)

Base.copy(x::SparseOperator) = SparseOperator(x.basis_l, x.basis_r, copy(x.data))
operators.dense(a::SparseOperator) = DenseOperator(a.basis_l, a.basis_r, Matrix(a.data))

"""
    sparse(op::Operator)

Convert an arbitrary operator into a [`SparseOperator`](@ref).
"""
sparse(a::Operator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(full(op)) instead."))
sparse(a::SparseOperator) = copy(a)
sparse(a::DenseOperator) = SparseOperator(a.basis_l, a.basis_r, sparse(a.data))

==(x::SparseOperator, y::SparseOperator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


# Arithmetic operations
+(a::SparseOperator, b::SparseOperator) = (check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data+b.data))
+(a::SparseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data+b.data))
+(a::DenseOperator, b::SparseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::SparseOperator) = SparseOperator(a.basis_l, a.basis_r, -a.data)
-(a::SparseOperator, b::SparseOperator) = (check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data-b.data))
-(a::SparseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data-b.data))
-(a::DenseOperator, b::SparseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data-b.data))

*(a::SparseOperator, b::SparseOperator) = (check_multiplicable(a, b); SparseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::SparseOperator) = SparseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, a.data/complex(b))

operators.dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')
operators.ishermitian(A::SparseOperator) = (A.basis_l == A.basis_r) && ishermitian(A.data)

operators.tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))
operators.tensor(a::DenseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))
operators.tensor(a::SparseOperator, b::DenseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

operators.tr(op::SparseOperator) = (check_samebases(op); tr(op.data))

operators.conj(op::SparseOperator) = SparseOperator(op.basis_l, op.basis_r, conj(op.data))
operators.conj!(op::SparseOperator) = conj!(op.data)

function operators.ptrace(op::SparseOperator, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    shape = [op.basis_l.shape; op.basis_r.shape]
    data = sparsematrix.ptrace(op.data, shape, indices)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    SparseOperator(b_l, b_r, data)
end

function operators.expect(op::SparseOperator, state::Ket)# where T <: Union{Ket, Bra}
    check_samebases(op.basis_r, state.basis)
    check_samebases(op.basis_l, state.basis)
    state.data' * op.data * state.data
end


function operators.expect(op::SparseOperator, state::DenseOperator)
    check_samebases(op.basis_r, state.basis_l)
    check_samebases(op.basis_l, state.basis_r)
    result = ComplexF64(0.)
    @inbounds for colindex = 1:op.data.n
        for i=op.data.colptr[colindex]:op.data.colptr[colindex+1]-1
            result += op.data.nzval[i]*state.data[colindex, op.data.rowval[i]]
        end
    end
    result
end

function operators.permutesystems(rho::SparseOperator, perm::Vector{Int})
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert isperm(perm)
    shape = [rho.basis_l.shape; rho.basis_r.shape]
    data = sparsematrix.permutedims(rho.data, shape, [perm; perm + length(perm)])
    SparseOperator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

operators.identityoperator(::Type{SparseOperator}, b1::Basis, b2::Basis) = SparseOperator(b1, b2, sparse(ComplexF64(1)*I, length(b1), length(b2)))
operators.identityoperator(b1::Basis, b2::Basis) = identityoperator(SparseOperator, b1, b2)
operators.identityoperator(b::Basis) = identityoperator(b, b)

"""
    diagonaloperator(b::Basis)

Create a diagonal operator of type [`SparseOperator`](@ref).
"""
function diagonaloperator(b::Basis, diag::Vector{T}) where T <: Number
  @assert 1 <= length(diag) <= prod(b.shape)
  SparseOperator(b, spdiagm(convert(Vector{ComplexF64}, diag)))
end


# Fast in-place multiplication implementations
operators.gemm!(alpha, M::SparseOperator, b::DenseOperator, beta, result::DenseOperator) = sparsematrix.gemm!(convert(ComplexF64, alpha), M.data, b.data, convert(ComplexF64, beta), result.data)
operators.gemm!(alpha, a::DenseOperator, M::SparseOperator, beta, result::DenseOperator) = sparsematrix.gemm!(convert(ComplexF64, alpha), a.data, M.data, convert(ComplexF64, beta), result.data)
operators.gemv!(alpha, M::SparseOperator, b::Ket, beta, result::Ket) = sparsematrix.gemv!(convert(ComplexF64, alpha), M.data, b.data, convert(ComplexF64, beta), result.data)
operators.gemv!(alpha, b::Bra, M::SparseOperator, beta, result::Bra) = sparsematrix.gemv!(convert(ComplexF64, alpha), b.data, M.data, convert(ComplexF64, beta), result.data)

end # module
