using Base.Test
using QuantumOptics

@testset "lazyoperators" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1 = NLevelBasis(4)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

b = b1⊗b2⊗b3

op1a = DenseOperator(b1, b1, rand(Complex128, length(b1), length(b1)))
op1b = DenseOperator(b1, b1, rand(Complex128, length(b1), length(b1)))
op2a = DenseOperator(b2, b2, rand(Complex128, length(b2), length(b2)))
op2b = DenseOperator(b2, b2, rand(Complex128, length(b2), length(b2)))
op3a = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))
op3b = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))
I1 = full(identityoperator(b1))
I2 = full(identityoperator(b2))
I3 = full(identityoperator(b3))

end # testset
