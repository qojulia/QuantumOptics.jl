module operators_lazysum

export LazySum

import Base: ==, *, /, +, -
import ..operators
import SparseArrays: sparse
import LinearAlgebra: mul!

using ..bases, ..states, ..operators, ..operators_dense
using SparseArrays, LinearAlgebra

"""
    LazySum([factors,] operators)

Lazy evaluation of sums of operators.

All operators have to be given in respect to the same bases. The field
`factors` accounts for an additional multiplicative factor for each operator
stored in the `operators` field.
"""
mutable struct LazySum{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factors::Vector{ComplexF64}
    operators::Vector{AbstractOperator}

    function LazySum(factors::Vector{ComplexF64}, operators::Vector{AbstractOperator})
        @assert length(operators)>0
        @assert length(operators)==length(factors)
        for i = 2:length(operators)
            @assert operators[1].basis_l == operators[i].basis_l
            @assert operators[1].basis_r == operators[i].basis_r
        end
        new{typeof(operators[1].basis_l),typeof(operators[1].basis_r)}(operators[1].basis_l, operators[1].basis_r, factors, operators)
    end
end
LazySum(factors::Vector{T}, operators::Vector) where {T<:Number} = LazySum(complex(factors), AbstractOperator[op for op in operators])
LazySum(operators::AbstractOperator...) = LazySum(ones(ComplexF64, length(operators)), AbstractOperator[operators...])

Base.copy(x::LazySum) = LazySum(copy(x.factors), [copy(op) for op in x.operators])

operators.dense(op::LazySum) = sum(op.factors .* dense.(op.operators))
SparseArrays.sparse(op::LazySum) = sum(op.factors .* sparse.(op.operators))

==(x::LazySum, y::LazySum) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factors==y.factors

# Arithmetic operations
+(a::LazySum, b::LazySum) = (check_samebases(a,b); LazySum([a.factors; b.factors], [a.operators; b.operators]))

-(a::LazySum) = LazySum(-a.factors, a.operators)
-(a::LazySum, b::LazySum) = (check_samebases(a,b); LazySum([a.factors; -b.factors], [a.operators; b.operators]))

*(a::LazySum, b::Number) = LazySum(b*a.factors, a.operators)
*(a::Number, b::LazySum) = LazySum(a*b.factors, b.operators)

/(a::LazySum, b::Number) = LazySum(a.factors/b, a.operators)

operators.dagger(op::LazySum) = LazySum(conj.(op.factors), dagger.(op.operators))

operators.tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

function operators.ptrace(op::LazySum, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    D = AbstractOperator[ptrace(op_i, indices) for op_i in op.operators]
    LazySum(op.factors, D)
end

operators.normalize!(op::LazySum) = (op.factors /= tr(op); nothing)

operators.permutesystems(op::LazySum, perm::Vector{Int}) = LazySum(op.factors, AbstractOperator[permutesystems(op_i, perm) for op_i in op.operators])

operators.identityoperator(::Type{LazySum}, b1::Basis, b2::Basis) = LazySum(identityoperator(b1, b2))


# Fast in-place multiplication
function mul!(result::Ket{BL}, a::LazySum{BL,BR}, b::Ket{BR},
            alpha::Number, beta::Number) where {BL<:Basis,BR<:Basis}
    mul!(result, a.operators[1], b, alpha*a.factors[1], beta)
    for i=2:length(a.operators)
        mul!(result, a.operators[i], b, alpha*a.factors[i], 1.0)
    end
end
mul!(result::Ket{BL}, a::LazySum{BL,BR}, b::Ket{BR}) where {BL<:Basis,BR<:Basis} =
    mul!(result, a, b, 1.0, 0.0)

function mul!(result::Bra{BR}, a::Bra{BL}, b::LazySum{BL,BR},
            alpha::Number, beta::Number) where {BR<:Basis,BL<:Basis}
    mul!(result, a, b.operators[1], alpha*b.factors[1], beta)
    for i=2:length(b.operators)
        mul!(result, a, b.operators[i], alpha*b.factors[i], 1.0)
    end
end
mul!(result::Bra{BR}, a::Bra{BL}, b::LazySum{BL,BR}) where {BR<:Basis,BL<:Basis} =
    mul!(result, a, b, 1.0, 0.0)

end # module
