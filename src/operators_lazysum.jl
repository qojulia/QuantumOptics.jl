module operators_lazysum

export LazySum

import Base: ==, *, /, +, -
import ..operators

using ..bases, ..states, ..operators, ..operators_dense


"""
    LazySum([factors,] operators)

Lazy evaluation of sums of operators.

All operators have to be given in respect to the same bases. The field
`factors` accounts for an additional multiplicative factor for each operator
stored in the `operators` field.
"""
type LazySum <: Operator
    basis_l::Basis
    basis_r::Basis
    factors::Vector{Complex128}
    operators::Vector{Operator}

    function LazySum(factors::Vector{Complex128}, operators::Vector{Operator})
        @assert length(operators)>0
        @assert length(operators)==length(factors)
        for i = 2:length(operators)
            @assert operators[1].basis_l == operators[i].basis_l
            @assert operators[1].basis_r == operators[i].basis_r
        end
        new(operators[1].basis_l, operators[1].basis_r, factors, operators)
    end
end
LazySum{T<:Number}(factors::Vector{T}, operators::Vector) = LazySum(complex(factors), Operator[op for op in operators])
LazySum(operators::Operator...) = LazySum(ones(Complex128, length(operators)), Operator[operators...])

Base.copy(x::LazySum) = LazySum(copy(x.factors), [copy(op) for op in x.operators])

Base.full(op::LazySum) = sum(op.factors .* full.(op.operators))
Base.sparse(op::LazySum) = sum(op.factors .* sparse.(op.operators))

==(x::LazySum, y::LazySum) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factors==y.factors

# Arithmetic operations
+(a::LazySum, b::LazySum) = (check_samebases(a,b); LazySum([a.factors; b.factors], [a.operators; b.operators]))

-(a::LazySum) = LazySum(-a.factors, a.operators)
-(a::LazySum, b::LazySum) = (check_samebases(a,b); LazySum([a.factors; -b.factors], [a.operators; b.operators]))

*(a::LazySum, b::Number) = LazySum(b*a.factors, a.operators)
*(a::Number, b::LazySum) = LazySum(a*b.factors, b.operators)

/(a::LazySum, b::Number) = LazySum(a.factors/b, a.operators)

operators.dagger(op::LazySum) = LazySum(conj.(op.factors), dagger.(op.operators))

operators.trace(op::LazySum) = sum(op.factors .* trace.(op.operators))

function operators.ptrace(op::LazySum, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    if rank==0
        return trace(op)
    end
    D = Operator[ptrace(op_i, indices) for op_i in op.operators]
    LazySum(op.factors, D)
end

operators.normalize!(op::LazySum) = (op.factors /= trace(op))

operators.permutesystems(op::LazySum, perm::Vector{Int}) = LazySum(op.factors, Operator[permutesystems(op_i, perm) for op_i in op.operators])

operators.identityoperator(::Type{LazySum}, b1::Basis, b2::Basis) = LazySum(identityoperator(b1, b2))


# Fast in-place multiplication
function operators.gemv!(alpha, a::LazySum, b::Ket, beta, result::Ket)
    operators.gemv!(alpha*a.factors[1], a.operators[1], b, beta, result)
    for i=2:length(a.operators)
        operators.gemv!(alpha*a.factors[i], a.operators[i], b, 1, result)
    end
end

function operators.gemv!(alpha, a::Bra, b::LazySum, beta, result::Bra)
    operators.gemv!(alpha*b.factors[1], a, b.operators[1], beta, result)
    for i=2:length(b.operators)
        operators.gemv!(alpha*b.factors[i], a, b.operators[i], 1, result)
    end
end

end # module
