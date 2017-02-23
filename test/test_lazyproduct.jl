using Base.Test
using QuantumOptics

@testset "lazyproduct" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1 = NLevelBasis(4)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

b = b1⊗b2⊗b3

op1 = DenseOperator(b1, b2, rand(Complex128, length(b1), length(b2)))
op2 = DenseOperator(b2, b3, rand(Complex128, length(b2), length(b3)))
op3 = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))

# LazyProduct
op_l = (0.4*lazy(op1))*(-lazy(op2)*lazy(op3)/3)
op = -0.4*op1*op2*op3/3

@test 1e-15 > D(op, full(op_l))
@test 1e-15 > D(op, sparse(op_l))

end # testset
