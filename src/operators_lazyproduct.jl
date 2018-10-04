module operators_lazyproduct

export LazyProduct

import Base: ==, *, /, +, -
import LinearAlgebra: mul!
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
mutable struct LazyProduct{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factor::ComplexF64
    operators::Vector{AbstractOperator}

    function LazyProduct(operators::Vector{AbstractOperator}, factor::Number=1)
        if length(operators) < 1
            throw(ArgumentError("LazyProduct needs at least one operator."))
        end
        for i = 2:length(operators)
            check_multiplicable(operators[i-1], operators[i])
        end
        new{typeof(operators[1].basis_l),typeof(operators[end].basis_r)}(operators[1].basis_l, operators[end].basis_r, factor, operators)
    end
end
LazyProduct(operators::Vector, factor::Number=1) = LazyProduct(convert(Vector{AbstractOperator}, operators), factor)
LazyProduct(operators::AbstractOperator...) = LazyProduct(AbstractOperator[operators...])

Base.copy(x::LazyProduct) = LazyProduct([copy(op) for op in x.operators], x.factor)

operators.dense(op::LazyProduct) = op.factor*prod(dense.(op.operators))
SparseArrays.sparse(op::LazyProduct) = op.factor*prod(sparse.(op.operators))

==(x::LazyProduct, y::LazyProduct) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor == y.factor


# Arithmetic operations
-(a::LazyProduct) = LazyProduct(a.operators, -a.factor)

*(a::LazyProduct, b::LazyProduct) = (check_multiplicable(a, b); LazyProduct([a.operators; b.operators], a.factor*b.factor))
*(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor*b)
*(a::Number, b::LazyProduct) = LazyProduct(b.operators, a*b.factor)

/(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor/b)


operators.dagger(op::LazyProduct) = LazyProduct(dagger.(reverse(op.operators)), conj(op.factor))

operators.tr(op::LazyProduct) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

operators.ptrace(op::LazyProduct, indices::Vector{Int}) = throw(ArgumentError("Partial trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. dense() or sparse()."))

operators.permutesystems(op::LazyProduct, perm::Vector{Int}) = LazyProduct(AbstractOperator[permutesystems(op_i, perm) for op_i in op.operators], op.factor)

operators.identityoperator(::Type{LazyProduct}, b1::Basis, b2::Basis) = LazyProduct(identityoperator(b1, b2))


# Fast in-place multiplication
function operators.gemv!(alpha::Number, a::LazyProduct, b::Ket, beta::Number, result::Ket)
    tmp1 = Ket(a.operators[end].basis_l)
    mul!(tmp1, a.operators[end], b, a.factor, 0.0)
    for i=length(a.operators)-1:-1:2
        tmp2 = Ket(a.operators[i].basis_l)
        mul!(tmp2, a.operators[i], tmp1)
        tmp1 = tmp2
    end
    mul!(result, a.operators[1], tmp1, alpha, beta)
end

function operators.gemv!(alpha::Number, a::Bra, b::LazyProduct, beta::Number, result::Bra)
    tmp1 = Bra(b.operators[1].basis_r)
    mul!(tmp1, a, b.operators[1], b.factor, 0.0)
    for i=2:length(b.operators)-1
        tmp2 = Bra(b.operators[i].basis_r)
        mul!(tmp2, tmp1, b.operators[i])
        tmp1 = tmp2
    end
    mul!(result, tmp1, b.operators[end], alpha, beta)
end

mul!(result::Ket{BL}, a::LazyProduct{BL,BR}, b::Ket{BR},
        alpha::Number, beta::Number) where {BL<:Basis, BR<:Basis} =
    operators.gemv!(alpha, a, b, beta, result)
mul!(result::Ket{BL}, a::LazyProduct{BL,BR}, b::Ket{BR}) where {BL<:Basis, BR<:Basis} =
    mul!(result, a, b, 1.0, 0.0)

mul!(result::Bra{BR}, a::Bra{BL}, b::LazyProduct{BL,BR},
        alpha::Number, beta::Number) where {BL<:Basis, BR<:Basis} =
    operators.gemv!(alpha, a, b, beta, result)
mul!(result::Bra{BR}, a::Bra{BL}, b::LazyProduct{BL,BR}) where {BL<:Basis, BR<:Basis} =
    mul!(result, a, b, 1.0, 0.0)

end # module
