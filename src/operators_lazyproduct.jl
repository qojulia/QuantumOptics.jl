"""
Lazy evaluation of product of operators.

The factors of the product are stored in the "operators" field. Additionally a
complex factor is stored in the "factor" field which allows for fast
multiplication with a number.
"""
type LazyProduct <: Operator
    basis_l::Basis
    basis_r::Basis
    factor::Complex128
    operators::Vector{Operator}

    function LazyProduct(operators::Vector{Operator}, factor::Number=1)
        @assert length(operators) > 0
        for i = 2:length(operators)
            @assert multiplicable(operators[i-1].basis_r, operators[i].basis_l)
        end
        new(operators[1].basis_l, operators[end].basis_r, factor, operators)
    end
end
LazyProduct(operators::Vector, factor::Number=1) = LazyProduct(Operator[op for op in operators], factor)
LazyProduct(operators::Operator...) = LazyProduct(Operator[operators...])

Base.full(op::LazyProduct) = op.factor*prod(full(op_i) for op_i in op.operators)
Base.sparse(op::LazyProduct) = op.factor*prod(sparse(op_i) for op_i in op.operators)

==(x::LazyProduct, y::LazyProduct) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor == y.factor

*(a::LazyProduct, b::LazyProduct) = LazyProduct([a.operators; b.operators], a.factor*b.factor)
*(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor*b)
*(a::Number, b::LazyProduct) = LazyProduct(b.operators, a*b.factor)
*(a::LazyWrapper, b::LazyWrapper) = LazyProduct([a.operator, b.operator], a.factor*b.factor)
*(a::LazyWrapper, b::LazyProduct) = LazyProduct([a.operator; b.operators], a.factor*b.factor)
*(a::LazyProduct, b::LazyWrapper) = LazyProduct([a.operators; b.operator], a.factor*b.factor)

/(a::LazyProduct, b::Number) = LazyProduct(a.operators, a.factor/b)

-(a::LazyProduct) = LazyProduct(a.operators, -a.factor)

identityoperator(::Type{LazyProduct}, b1::Basis, b2::Basis) = LazyProduct(identityoperator(b1, b2))

dagger(op::LazyProduct) = LazyProduct([dagger(op_i) for op_i in reverse(op.operators)], conj(op.factor))

trace(op::LazyProduct) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. full() or sparse()."))

ptrace(op::LazyProduct, indices::Vector{Int}) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. full() or sparse()."))

operators.permutesystems(op::LazyProduct, perm::Vector{Int}) = LazyProduct(Operator[permutesystems(op_i, perm) for op_i in op.operators], op.factor)

function operators.gemv!(alpha, a::LazyProduct, b::Ket, beta, result::Ket)
    tmp1 = Ket(a.operators[end].basis_l)
    operators.gemv!(Complex(1.)*a.factor, a.operators[end], b, Complex(0.), tmp1)
    for i=length(a.operators)-1:-1:2
        tmp2 = Ket(a.operators[i].basis_l)
        operators.gemv!(Complex(1.), a.operators[i], tmp1, Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, a.operators[1], tmp1, beta, result)
end

function operators.gemv!(alpha, a::Bra, b::LazyProduct, beta, result::Bra)
    tmp1 = Bra(b.operators[1].basis_r)
    operators.gemv!(Complex(1.)*b.factor, a, b.operators[1], Complex(0.), tmp1)
    for i=2:length(b.operators)-1
        tmp2 = Bra(b.operators[i].basis_r)
        operators.gemv!(Complex(1.), tmp1, b.operators[i], Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, tmp1, b.operators[end], beta, result)
end
