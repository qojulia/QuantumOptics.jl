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

# LazyTensor
# op = LazyTensor(b, Dict(1=>op1a, 2=>op2a, 3=>op3a))
op = lazy(op1a) ⊗ lazy(op2a) ⊗ lazy(op3a)
@test 1e-15 > D(op1a⊗op2a⊗op3a, op)
@test 1e-15 > D(op1a⊗op2a⊗op3a, sparse(op))

op = 0.3*lazy(op1a) ⊗ (-lazy(op2a)) ⊗ lazy(op3a)/4
@test 1e-15 > D(-0.3*op1a⊗op2a⊗op3a/4, full(op))
@test 1e-15 > D(-0.3*op1a⊗op2a⊗op3a/4, sparse(op))

op = LazyTensor(b, Dict(1=>sparse(op1a), 2=>op2a, 3=>sparse(op3a)))
@test 1e-15 > D(op1a⊗op2a⊗op3a, full(op))
@test 1e-15 > D(op1a⊗op2a⊗op3a, sparse(op))

op = LazyTensor(b, Dict(1=>sparse(op1a), 3=>op3a))
@test 1e-15 > D(op1a⊗I2⊗op3a, full(op))
@test 1e-15 > D(op1a⊗I2⊗op3a, sparse(op))

op = LazyTensor(b, [1,3], [sparse(op1a), op3a])
@test 1e-15 > D(op1a⊗I2⊗op3a, full(op))
@test 1e-15 > D(op1a⊗I2⊗op3a, sparse(op))

op = LazyTensor(b, Dict(1=>op1a))
@test 1e-15 > D(op1a⊗I2⊗I3, full(op))
@test 1e-15 > D(op1a⊗I2⊗I3, sparse(op))

op = LazyTensor(b, Dict(2=>op2a))
@test 1e-15 > D(I1⊗op2a⊗I3, full(op))
@test 1e-15 > D(I1⊗op2a⊗I3, sparse(op))

op = LazyTensor(b, Dict(3=>op3a))
@test 1e-15 > D(I1⊗I2⊗op3a, full(op))
@test 1e-15 > D(I1⊗I2⊗op3a, sparse(op))

x = LazyTensor(b, Dict(3=>op3a))
y = LazyTensor(b, [1,3], [sparse(op1b), op3b])
@test 1e-15 > D(op1b⊗I2⊗(op3a*op3b), x*y)

x = LazyTensor(b, [2,1,3], [sparse(op2b), op1b, op3b])
y = LazyTensor(b, Dict(2=>op2a))
@test 1e-15 > D(op1b⊗(op2b*op2a)⊗op3b, x*y)

end # testset
