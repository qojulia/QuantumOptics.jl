using Base.Test
using QuantumOptics

srand(0)

b1a = NLevelBasis(2)
b1b = SpinBasis(3//2)
b2a = SpinBasis(1//2)
b2b = FockBasis(7)
b3a = FockBasis(2)
b3b = NLevelBasis(4)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

function test_op_equal(op1, op2)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) 1e-11
end

# DenseOperator, SparseOperator and LazyTensor
op1 = DenseOperator(b1a, b1b, rand(Complex128, length(b1a), length(b1b)))
op2 = DenseOperator(b2a, b2b, rand(Complex128, length(b2a), length(b2b)))
op3 = DenseOperator(b3a, b3b, rand(Complex128, length(b3a), length(b3b)))
op123 = LazyTensor(b_l, b_r, [1, 2, 3], [op1, op2, op3])

op132 = op1⊗op3⊗op2
test_op_equal(permutesystems(op123, [1, 3, 2]), op132)
test_op_equal(permutesystems(full(op123), [1, 3, 2]), op132)
test_op_equal(permutesystems(sparse(op123), [1, 3, 2]), op132)

op213 = op2⊗op1⊗op3
test_op_equal(permutesystems(op123, [2, 1, 3]), op213)
test_op_equal(permutesystems(full(op123), [2, 1, 3]), op213)
test_op_equal(permutesystems(sparse(op123), [2, 1, 3]), op213)

op231 = op2⊗op3⊗op1
test_op_equal(permutesystems(op123, [2, 3, 1]), op231)
test_op_equal(permutesystems(full(op123), [2, 3, 1]), op231)
test_op_equal(permutesystems(sparse(op123), [2, 3, 1]), op231)

op312 = op3⊗op1⊗op2
test_op_equal(permutesystems(op123, [3, 1, 2]), op312)
test_op_equal(permutesystems(full(op123), [3, 1, 2]), op312)
test_op_equal(permutesystems(sparse(op123), [3, 1, 2]), op312)

op321 = op3⊗op2⊗op1
test_op_equal(permutesystems(op123, [3, 2, 1]), op321)
test_op_equal(permutesystems(full(op123), [3, 2, 1]), op321)
test_op_equal(permutesystems(sparse(op123), [3, 2, 1]), op321)


# LazyProduct
op1 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op2 = DenseOperator(b_r, b_l, rand(Complex128, length(b_r), length(b_l)))
op3 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op123 = LazyProduct(op1, op2, op3)

test_op_equal(permutesystems(op123, [1, 3, 2]), permutesystems(full(op123), [1, 3, 2]))
test_op_equal(permutesystems(op123, [2, 1, 3]), permutesystems(full(op123), [2, 1, 3]))
test_op_equal(permutesystems(op123, [2, 3, 1]), permutesystems(full(op123), [2, 3, 1]))
test_op_equal(permutesystems(op123, [3, 2, 1]), permutesystems(full(op123), [3, 2, 1]))
test_op_equal(permutesystems(op123, [3, 1, 2]), permutesystems(full(op123), [3, 1, 2]))


# LazySum
op1 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op2 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op3 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op123 = LazySum(op1, op2, op3)

test_op_equal(permutesystems(op123, [1, 3, 2]), permutesystems(full(op123), [1, 3, 2]))
test_op_equal(permutesystems(op123, [2, 1, 3]), permutesystems(full(op123), [2, 1, 3]))
test_op_equal(permutesystems(op123, [2, 3, 1]), permutesystems(full(op123), [2, 3, 1]))
test_op_equal(permutesystems(op123, [3, 2, 1]), permutesystems(full(op123), [3, 2, 1]))
test_op_equal(permutesystems(op123, [3, 1, 2]), permutesystems(full(op123), [3, 1, 2]))
