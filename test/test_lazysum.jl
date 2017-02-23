using Base.Test
using QuantumOptics

@testset "lazysum" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1 = NLevelBasis(4)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

b = b1⊗b2⊗b3

op1 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))
op2 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))
op3 = DenseOperator(b, b, rand(Complex128, length(b), length(b)))


# Arithmetic
op_l = (-0.2*lazy(op1)) - 0.5*lazy(op2) + lazy(sparse(op3))/3
op = -0.2*op1 - 0.5 * op2 + op3/3

@test 1e-14 > D(op, full(op_l))
@test 1e-14 > D(op, sparse(op_l))

op_l = (2*(lazy(sparse(op1)) - lazy(op2)) + lazy(op3))/3
op = (2*(op1 - op2) + op3)/3

@test 1e-14 > D(op, full(op_l))
@test 1e-14 > D(op, sparse(op_l))

end # testset
