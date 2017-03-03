using Base.Test
using QuantumOptics

@testset "identityoperator" begin

b1 = NLevelBasis(2)
b2 = SpinBasis(3//2)
b3 = FockBasis(2)
b = b1⊗b2⊗b3

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

Idense = identityoperator(DenseOperator, b)
Isparse = identityoperator(SparseOperator, b)
Iltensor = identityoperator(LazyTensor, b)
Ilsum = identityoperator(LazySum, b)
Ilprod = identityoperator(LazyProduct, b)

@test typeof(Idense) == DenseOperator
@test typeof(Isparse) == SparseOperator
@test typeof(Iltensor) == LazyTensor
@test typeof(Ilsum) == LazySum
@test typeof(Ilprod) == LazyProduct

@test full(Isparse) == Idense
@test full(Iltensor) == Idense
@test full(Ilsum) == Idense
@test full(Ilprod) == Idense

@test Isparse == identityoperator(b1) ⊗ identityoperator(b2) ⊗ identityoperator(b3)
@test Idense == identityoperator(DenseOperator, b1) ⊗ identityoperator(DenseOperator, b2) ⊗ identityoperator(DenseOperator, b3)
identityoperator(LazyTensor, b1) ⊗ identityoperator(LazyTensor, b2) ⊗ identityoperator(LazyTensor, b3)
@test Iltensor == identityoperator(LazyTensor, b1) ⊗ identityoperator(LazyTensor, b2) ⊗ identityoperator(LazyTensor, b3)

op = DenseOperator(b, rand(Complex128, length(b), length(b)))

@test 1e-15 > D(op, identityoperator(op)*op)
@test 1e-15 > D(op, op*identityoperator(op))

op = sparse(op)

@test 1e-15 > D(op, identityoperator(op)*op)
@test 1e-15 > D(op, op*identityoperator(op))

op1 = DenseOperator(b1, rand(Complex128, length(b1), length(b1)))
op2 = DenseOperator(b2, rand(Complex128, length(b2), length(b2)))
op3 = DenseOperator(b3, rand(Complex128, length(b3), length(b3)))
op = LazyTensor(b, [1, 2, 3], [op1, op2, op3])

@test 1e-15 > D(op, identityoperator(op)*op)
@test 1e-15 > D(op, op*identityoperator(op))

end # testset
