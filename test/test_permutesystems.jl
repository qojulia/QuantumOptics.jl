using Base.Test
using QuantumOptics

@testset "permutesystems" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

b1a = NLevelBasis(2)
b1b = SpinBasis(3//2)
b2a = SpinBasis(1//2)
b2b = FockBasis(7)
b3a = FockBasis(2)
b3b = NLevelBasis(4)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b


# DenseOperator, SparseOperator and LazyTensor
op1 = DenseOperator(b1a, b1b, rand(Complex128, length(b1a), length(b1b)))
op2 = DenseOperator(b2a, b2b, rand(Complex128, length(b2a), length(b2b)))
op3 = DenseOperator(b3a, b3b, rand(Complex128, length(b3a), length(b3b)))
op123 = LazyTensor(b_l, b_r, [1, 2, 3], [op1, op2, op3])

op132 = op1⊗op3⊗op2
@test 1e-14 > D(permutesystems(op123, [1, 3, 2]), op132)
@test 1e-14 > D(permutesystems(full(op123), [1, 3, 2]), op132)
@test 1e-14 > D(permutesystems(sparse(op123), [1, 3, 2]), op132)

op213 = op2⊗op1⊗op3
@test 1e-14 > D(permutesystems(op123, [2, 1, 3]), op213)
@test 1e-14 > D(permutesystems(full(op123), [2, 1, 3]), op213)
@test 1e-14 > D(permutesystems(sparse(op123), [2, 1, 3]), op213)

op231 = op2⊗op3⊗op1
@test 1e-14 > D(permutesystems(op123, [2, 3, 1]), op231)
@test 1e-14 > D(permutesystems(full(op123), [2, 3, 1]), op231)
@test 1e-14 > D(permutesystems(sparse(op123), [2, 3, 1]), op231)

op312 = op3⊗op1⊗op2
@test 1e-14 > D(permutesystems(op123, [3, 1, 2]), op312)
@test 1e-14 > D(permutesystems(full(op123), [3, 1, 2]), op312)
@test 1e-14 > D(permutesystems(sparse(op123), [3, 1, 2]), op312)

op321 = op3⊗op2⊗op1
@test 1e-14 > D(permutesystems(op123, [3, 2, 1]), op321)
@test 1e-14 > D(permutesystems(full(op123), [3, 2, 1]), op321)
@test 1e-14 > D(permutesystems(sparse(op123), [3, 2, 1]), op321)


# LazyProduct
op1 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op2 = DenseOperator(b_r, b_l, rand(Complex128, length(b_r), length(b_l)))
op3 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op123 = LazyProduct(op1, op2, op3)

@test 1e-11 > D(permutesystems(op123, [1, 3, 2]), permutesystems(full(op123), [1, 3, 2]))
@test 1e-11 > D(permutesystems(op123, [2, 1, 3]), permutesystems(full(op123), [2, 1, 3]))
@test 1e-11 > D(permutesystems(op123, [2, 3, 1]), permutesystems(full(op123), [2, 3, 1]))
@test 1e-11 > D(permutesystems(op123, [3, 2, 1]), permutesystems(full(op123), [3, 2, 1]))
@test 1e-11 > D(permutesystems(op123, [3, 1, 2]), permutesystems(full(op123), [3, 1, 2]))


# LazySum
op1 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op2 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op3 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op123 = LazySum(op1, op2, op3)

@test 1e-14 > D(permutesystems(op123, [1, 3, 2]), permutesystems(full(op123), [1, 3, 2]))
@test 1e-14 > D(permutesystems(op123, [2, 1, 3]), permutesystems(full(op123), [2, 1, 3]))
@test 1e-14 > D(permutesystems(op123, [2, 3, 1]), permutesystems(full(op123), [2, 3, 1]))
@test 1e-14 > D(permutesystems(op123, [3, 2, 1]), permutesystems(full(op123), [3, 2, 1]))
@test 1e-14 > D(permutesystems(op123, [3, 1, 2]), permutesystems(full(op123), [3, 1, 2]))

end # testset
