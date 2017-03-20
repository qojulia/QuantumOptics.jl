module operators_sparse

import Base: ==, *, /, +, -, ishermitian
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems

using ..bases, ..states, ..operators, ..operators_dense, ..sparsematrix

export SparseOperator, diagonaloperator


"""
Sparse array implementation of Operator.

The matrix is stored as the julia built-in type SparseMatrixCSC
in the data field.
"""
type SparseOperator <: Operator
    basis_l::Basis
    basis_r::Basis
    data::SparseMatrixCSC{Complex128}
    function SparseOperator(b1::Basis, b2::Basis, data)
        if length(b1) != size(data, 1) || length(b2) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(b1, b2, data)
    end
end

SparseOperator(b::Basis, data::SparseMatrixCSC{Complex128}) = SparseOperator(b, b, data)
SparseOperator(b::Basis, data::Matrix{Complex128}) = SparseOperator(b, sparse(data))
SparseOperator(op::DenseOperator) = SparseOperator(op.basis_l, op.basis_r, sparse(op.data))

SparseOperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, spzeros(Complex128, length(b1), length(b2)))
SparseOperator(b::Basis) = SparseOperator(b, b)

Base.full(a::SparseOperator) = DenseOperator(a.basis_l, a.basis_r, full(a.data))

Base.sparse(a::Operator) = throw(ArgumentError("Direct conversion from $(typeof(a)) not implemented. Use sparse(full(op)) instead."))
Base.sparse(a::SparseOperator) = deepcopy(a)
Base.sparse(a::DenseOperator) = SparseOperator(a.basis_l, a.basis_r, sparse(a.data))

==(x::SparseOperator, y::SparseOperator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


# Arithmetic operations
*(a::SparseOperator, b::SparseOperator) = (check_multiplicable(a.basis_r, b.basis_l); SparseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::SparseOperator) = SparseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::SparseOperator, b::SparseOperator) = (operators.check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::SparseOperator) = SparseOperator(a.basis_l, a.basis_r, -a.data)
-(a::SparseOperator, b::SparseOperator) = (operators.check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data-b.data))


dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')

identityoperator(::Type{SparseOperator}, b1::Basis, b2::Basis) = SparseOperator(b1, b2, speye(Complex128, length(b1), length(b2)))
identityoperator(b1::Basis, b2::Basis) = identityoperator(SparseOperator, b1, b2)
identityoperator(b::Basis) = identityoperator(b, b)

trace(op::SparseOperator) = trace(op.data)

function operators.ptrace(op::SparseOperator, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    if rank==0
        return trace(op)
    end
    shape = [reverse(op.basis_l.shape); reverse(op.basis_r.shape)]
    indices_data = length(op.basis_l.shape) - indices + 1
    data = sparsematrix.ptrace(op.data, shape, indices_data)
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    SparseOperator(b_l, b_r, data)
end

function operators.expect(op::SparseOperator, state::DenseOperator)
    bases.check_equal(op.basis_r, state.basis_l)
    bases.check_equal(op.basis_l, state.basis_r)
    result = Complex128(0.)
    @inbounds for colindex = 1:op.data.n
        for i=op.data.colptr[colindex]:op.data.colptr[colindex+1]-1
            result += op.data.nzval[i]*state.data[colindex, op.data.rowval[i]]
        end
    end
    result
end

tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(a.data, b.data))

function permutesystems(rho::SparseOperator, perm::Vector{Int})
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert isperm(perm)
    shape = [reverse(rho.basis_l.shape); reverse(rho.basis_r.shape)]
    dataperm = length(perm) - reverse(perm) + 1
    data = sparsematrix.permutedims(rho.data, shape, [dataperm; dataperm + length(perm)])
    SparseOperator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

# Fast in-place multiplication implementations
operators.gemm!{T<:Complex}(alpha::T, M::SparseOperator, b::DenseOperator, beta::T, result::DenseOperator) = sparsematrix.gemm!(alpha, M.data, b.data, beta, result.data)
operators.gemm!{T<:Complex}(alpha::T, a::DenseOperator, M::SparseOperator, beta::T, result::DenseOperator) = sparsematrix.gemm!(alpha, a.data, M.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, M::SparseOperator, b::Ket, beta::T, result::Ket) = sparsematrix.gemv!(alpha, M.data, b.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, b::Bra, M::SparseOperator, beta::T, result::Bra) = sparsematrix.gemv!(alpha, b.data, M.data, beta, result.data)


"""
Diagonal operator.
"""
function diagonaloperator{T <: Number}(b::Basis, diag::Vector{T})
  @assert 1 <= length(diag) <= b.shape[1]
  SparseOperator(b, spdiagm(complex(float(diag))))
end


"""
Check if an operator is Hermitian.
"""
ishermitian(A::SparseOperator) = ishermitian(A.data)

end # module
