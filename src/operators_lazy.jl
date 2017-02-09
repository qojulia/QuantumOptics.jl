module operators_lazy

import Base: ==, *, /, +, -
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems

using Base.Cartesian
using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export lazy, LazyWrapper, LazyTensor, LazySum, LazyProduct


type LazyWrapper <: Operator
    basis_l::Basis
    basis_r::Basis
    factor::Complex128
    operator::Operator
    LazyWrapper(op::Operator, factor::Number=1) = new(op.basis_l, op.basis_r, factor, op)
end
lazy(op::LazyWrapper) = op
lazy(op::Operator) = LazyWrapper(op)

Base.full(op::LazyWrapper) = op.factor*full(op.operator)
Base.sparse(op::LazyWrapper) = op.factor*spare(op.operator)

==(x::LazyWrapper, y::LazyWrapper) = x.factor==y.factor && x.operator == y.operator

*(a::Number, b::LazyWrapper) = LazyWrapper(b.operator, a*b.factor)
*(a::LazyWrapper, b::Number) = LazyWrapper(a.operator, a.factor*b)

/(a::LazyWrapper, b::Number) = LazyWrapper(a.operator, a.factor/b)

-(a::LazyWrapper) = LazyWrapper(a.operator, -a.factor)

dagger(op::LazyWrapper) = LazyWrapper(dagger(op.operator), conj(op.factor))

trace(op::LazyWrapper) = trace(op.operator)*op.factor

ptrace(op::LazyWrapper, indices::Vector{Int}) = LazyWrapper(ptrace(op.operator, indices), op.factor)

permutesystems(op::LazyWrapper, perm::Vector{Int}) = LazyWrapper(permutesystems(op.operator, perm), op.factor)


include("operators_lazysum.jl")
include("operators_lazyproduct.jl")
include("operators_lazytensor.jl")

end # module
