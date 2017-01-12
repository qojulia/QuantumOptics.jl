module operators_sparse

import Base: ==, *, /, +, -
import ..operators

using ..bases, ..states, ..operators, ..sparsematrix

export SparseOperator, sparse_identityoperator, diagonaloperator


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


*(a::SparseOperator, b::SparseOperator) = (check_multiplicable(a.basis_r, b.basis_l); SparseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::SparseOperator) = SparseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::SparseOperator, b::Number) = SparseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::SparseOperator, b::SparseOperator) = (operators.check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::SparseOperator) = SparseOperator(a.basis_l, a.basis_r, -a.data)
-(a::SparseOperator, b::SparseOperator) = (operators.check_samebases(a,b); SparseOperator(a.basis_l, a.basis_r, a.data-b.data))

# Fast in-place multiplication implementations
operators.gemm!{T<:Complex}(alpha::T, M::SparseOperator, b::DenseOperator, beta::T, result::DenseOperator) = sparsematrix.gemm!(alpha, M.data, b.data, beta, result.data)
operators.gemm!{T<:Complex}(alpha::T, a::DenseOperator, M::SparseOperator, beta::T, result::DenseOperator) = sparsematrix.gemm!(alpha, a.data, M.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, M::SparseOperator, b::Ket, beta::T, result::Ket) = sparsematrix.gemv!(alpha, M.data, b.data, beta, result.data)
operators.gemv!{T<:Complex}(alpha::T, b::Bra, M::SparseOperator, beta::T, result::Bra) = sparsematrix.gemv!(alpha, b.data, M.data, beta, result.data)


operators.tensor(a::SparseOperator, b::SparseOperator) = SparseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(a.data, b.data))

operators.dagger(x::SparseOperator) = SparseOperator(x.basis_r, x.basis_l, x.data')

operators.norm(op::SparseOperator, p) = norm(op.data, p)
operators.trace(op::SparseOperator) = trace(op.data)

sparse_identityoperator(b::Basis) = SparseOperator(b, b, speye(Complex128, length(b)))
sparse_identityoperator(b1::Basis, b2::Basis) = SparseOperator(b1, b2, speye(Complex128, length(b1), length(b2)))

operators.identityoperator(b::Basis) = sparse_identityoperator(b)
operators.identityoperator(b1::Basis, b2::Basis) = sparse_identityoperator(b1, b2)
operators.identityoperator(op::SparseOperator) = sparse_identityoperator(op.basis_l, op.basis_r)


function operators.embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{SparseOperator})
    @assert length(indices) == length(operators)
    @assert length(basis.bases) >= maximum(indices)
    if length(basis.bases) == 1
        return SparseOperator(basis, basis, deepcopy(operators[1].data))
    end
    op_total = (1 in indices ? operators[findfirst(indices, 1)] : sparse_identityoperator(basis.bases[1]))
    for i=2:length(basis.bases)
        op = (i in indices ? operators[findfirst(indices, i)] : sparse_identityoperator(basis.bases[i]))
        op_total = tensor(op_total, op)
    end
    return op_total
end


"""
Diagonal operator.
"""
function diagonaloperator{T <: Number}(b::Basis, diag::Vector{T})
  @assert 1 <= length(diag) <= b.shape[1]
  SparseOperator(b, spdiagm(complex(float(diag))))
end

end # module
