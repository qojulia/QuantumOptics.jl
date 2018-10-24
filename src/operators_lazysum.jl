module operators_lazysum

export LazySum

import Base: ==, *, /, +, -
import ..operators
import SparseArrays: sparse

using ..bases, ..states, ..operators, ..operators_dense
using SparseArrays, LinearAlgebra

"""
    LazySum([factors,] operators)

Lazy evaluation of sums of operators.

All operators have to be given in respect to the same bases. The field
`factors` accounts for an additional multiplicative factor for each operator
stored in the `operators` field.
"""
mutable struct LazySum{BL<:Basis,BR<:Basis,T<:Tuple{Vararg{AbstractOperator{BL,BR}}}} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    factors::Vector{ComplexF64}
    operators::T
    function LazySum{BL,BR,T}(factors::Vector{ComplexF64}, operators::T) where {BL<:Basis,BR<:Basis,T<:Tuple{Vararg{AbstractOperator{BL,BR}}}}
        @assert length(operators)==length(factors)
        new(operators[1].basis_l, operators[1].basis_r, factors, operators)
    end
end
function LazySum(factors::Vector{ComplexF64}, operators::T) where {BL<:Basis,BR<:Basis,T<:Tuple{Vararg{AbstractOperator{BL,BR}}}}
    for i = 2:length(operators)
        operators[1].basis_l == operators[i].basis_l || throw(bases.IncompatibleBases())
        operators[1].basis_r == operators[i].basis_r || throw(bases.IncompatibleBases())
    end
    LazySum{BL,BR,T}(factors, operators)
end
LazySum(operators::AbstractOperator...) = LazySum(ones(ComplexF64, length(operators)), (operators...,))
LazySum(factors::Vector{T}, operators::Vector{T2}) where {T<:Number,B1<:Basis,B2<:Basis,T2<:AbstractOperator{B1,B2}} = LazySum(complex(factors), (operators...,))
LazySum() = throw(ArgumentError("LazySum needs at least one operator!"))

Base.copy(x::T) where T<:LazySum = T(copy(x.factors), ([copy(op) for op in x.operators]...,))

operators.dense(op::LazySum) = sum(op.factors .* dense.(op.operators))
operators.dense(op::LazySum{B1,B2,T}) where {B1<:Basis,B2<:Basis,T<:Tuple{AbstractOperator{B1,B2}}} = op.factors[1] * dense(op.operators[1])
SparseArrays.sparse(op::LazySum) = sum(op.factors .* sparse.(op.operators))
SparseArrays.sparse(op::LazySum{B1,B2,T}) where {B1<:Basis,B2<:Basis,T<:Tuple{AbstractOperator{B1,B2}}} = op.factors[1] * sparse(op.operators[1])

==(x::T, y::T) where T<:LazySum = (x.operators==y.operators && x.factors==y.factors)
==(x::LazySum, y::LazySum) = false

# Arithmetic operations
+(a::LazySum{B1,B2,T1}, b::LazySum{B1,B2,T2}) where {B1<:Basis,B2<:Basis,T1<:Tuple{Vararg{AbstractOperator{B1,B2}}},T2<:Tuple{Vararg{AbstractOperator{B1,B2}}}} = LazySum([a.factors; b.factors], (a.operators..., b.operators...))
+(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1<:Basis,B2<:Basis,B3<:Basis,B4<:Basis} = throw(bases.IncompatibleBases())

-(a::T) where T<:LazySum = T(-a.factors, a.operators)
-(a::LazySum{B1,B2,T1}, b::LazySum{B1,B2,T2}) where {B1<:Basis,B2<:Basis,T1<:Tuple{Vararg{AbstractOperator{B1,B2}}},T2<:Tuple{Vararg{AbstractOperator{B1,B2}}}} = LazySum([a.factors; -b.factors], (a.operators..., b.operators...))
-(a::LazySum{B1,B2}, b::LazySum{B3,B4}) where {B1<:Basis,B2<:Basis,B3<:Basis,B4<:Basis} = throw(bases.IncompatibleBases())

*(a::LazySum, b::Number) = LazySum(b*a.factors, a.operators)
*(a::Number, b::LazySum) = LazySum(a*b.factors, b.operators)

/(a::LazySum, b::Number) = LazySum(a.factors/b, a.operators)

operators.dagger(op::LazySum) = LazySum(conj.(op.factors), dagger.(op.operators))

operators.tr(op::LazySum) = sum(op.factors .* tr.(op.operators))

function operators.ptrace(op::LazySum, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    D = ([ptrace(op_i, indices) for op_i in op.operators]...,)
    LazySum(op.factors, D)
end

operators.normalize!(op::LazySum) = (op.factors /= tr(op); nothing)

operators.permutesystems(op::LazySum, perm::Vector{Int}) = LazySum(op.factors, ([permutesystems(op_i, perm) for op_i in op.operators]...,))

operators.identityoperator(::Type{LazySum}, b1::Basis, b2::Basis) = LazySum(identityoperator(b1, b2))


# Fast in-place multiplication
function operators.gemv!(alpha, a::LazySum{B1,B2}, b::Ket{B2}, beta, result::Ket{B1}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    operators.gemv!(alpha*a.factors[1], a.operators[1], b, beta, result)
    for i=2:length(a.operators)
        operators.gemv!(alpha*a.factors[i], a.operators[i], b, 1, result)
    end
end

function operators.gemv!(alpha, a::Bra{B1}, b::LazySum{B1,B2}, beta, result::Bra{B2}) where {B1<:Basis,B2<:Basis,B3<:Basis}
    operators.gemv!(alpha*b.factors[1], a, b.operators[1], beta, result)
    for i=2:length(b.operators)
        operators.gemv!(alpha*b.factors[i], a, b.operators[i], 1, result)
    end
end

end # module
