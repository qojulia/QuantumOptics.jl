module operators_lazyproduct

export LazyProduct

import Base: ==, *, /, +, -
import ..operators

using ..bases, ..states, ..operators, ..operators_dense
using SparseArrays


"""
    LazyProduct(operators[, factor=1])
    LazyProduct(op1, op2...)

Lazy evaluation of products of operators.

The factors of the product are stored in the `operators` field. Additionally a
complex factor is stored in the `factor` field which allows for fast
multiplication with numbers.
"""
mutable struct LazyProduct{BL<:Basis,BR<:Basis,T<:Tuple{Vararg{AbstractOperator}}} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factor::ComplexF64
    operators::T

    function LazyProduct{BL,BR,T}(operators::T, factor::Number=1) where {BL<:Basis,BR<:Basis,T<:Tuple{Vararg{AbstractOperator}}}
        for i = 2:length(operators)
            check_multiplicable(operators[i-1], operators[i])
        end
        new(operators[1].basis_l, operators[end].basis_r, factor, operators)
    end
end
function LazyProduct(operators::T, factor::Number=1) where {T<:Tuple{Vararg{AbstractOperator}}}
    BL = typeof(operators[1].basis_l)
    BR = typeof(operators[end].basis_r)
    LazyProduct{BL,BR,T}(operators, factor)
end
LazyProduct(operators::Vector{T}, factor::Number=1) where T<:AbstractOperator = LazyProduct((operators...,), factor)
LazyProduct(operators::AbstractOperator...) = LazyProduct((operators...,))
LazyProduct() = throw(ArgumentError("LazyProduct needs at least one operator!"))

Base.copy(x::T) where T<:LazyProduct = T(([copy(op) for op in x.operators]...,), x.factor)

operators.dense(op::LazyProduct) = op.factor*prod(dense.(op.operators))
operators.dense(op::LazyProduct{B1,B2,T}) where {B1<:Basis,B2<:Basis,T<:Tuple{AbstractOperator}} = op.factor*dense(op.operators[1])
SparseArrays.sparse(op::LazyProduct) = op.factor*prod(sparse.(op.operators))
SparseArrays.sparse(op::LazyProduct{B1,B2,T}) where {B1<:Basis,B2<:Basis,T<:Tuple{AbstractOperator}} = op.factor*sparse(op.operators[1])

==(x::LazyProduct, y::LazyProduct) = false
==(x::T, y::T) where T<:LazyProduct = (x.operators==y.operators && x.factor == y.factor)

# Arithmetic operations
-(a::T) where T<:LazyProduct = T(a.operators, -a.factor)

*(a::LazyProduct{B1,B2}, b::LazyProduct{B2,B3}) where {B1<:Basis,B2<:Basis,B3<:Basis} = LazyProduct((a.operators..., b.operators...), a.factor*b.factor)
*(a::T, b::Number) where T<:LazyProduct = T(a.operators, a.factor*b)
*(a::Number, b::T) where T<:LazyProduct = T(b.operators, a*b.factor)

/(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor/b)


operators.dagger(op::LazyProduct) = LazyProduct(dagger.(reverse(op.operators)), conj(op.factor))

operators.tr(op::LazyProduct) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

operators.ptrace(op::LazyProduct, indices::Vector{Int}) = throw(ArgumentError("Partial trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

operators.permutesystems(op::LazyProduct, perm::Vector{Int}) = LazyProduct(([permutesystems(op_i, perm) for op_i in op.operators]...,), op.factor)

operators.identityoperator(::Type{LazyProduct}, b1::Basis, b2::Basis) = LazyProduct(identityoperator(b1, b2))


# Fast in-place multiplication
function operators.gemv!(alpha, a::LazyProduct{B1,B2}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis}
    tmp1 = Ket(a.operators[end].basis_l)
    operators.gemv!(a.factor, a.operators[end], b, 0, tmp1)
    for i=length(a.operators)-1:-1:2
        tmp2 = Ket(a.operators[i].basis_l)
        operators.gemv!(1, a.operators[i], tmp1, 0, tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, a.operators[1], tmp1, beta, result)
end

function operators.gemv!(alpha, a::Bra{B1}, b::LazyProduct{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis}
    tmp1 = Bra(b.operators[1].basis_r)
    operators.gemv!(b.factor, a, b.operators[1], 0, tmp1)
    for i=2:length(b.operators)-1
        tmp2 = Bra(b.operators[i].basis_r)
        operators.gemv!(1, tmp1, b.operators[i], 0, tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, tmp1, b.operators[end], beta, result)
end

end # module
