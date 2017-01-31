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

function test_state_equal(x1, x2)
    @test_approx_eq_eps 0. norm(x1-x2) 1e-11
end

function test_op_equal(op1, op2)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) 1e-11
end

x1 = Ket(b_r, rand(Complex128, length(b_r)))
x2 = Ket(b_r, rand(Complex128, length(b_r)))
x3 = Ket(b_r, rand(Complex128, length(b_r)))

op1 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op2 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))
op3 = DenseOperator(b_l, b_r, rand(Complex128, length(b_l), length(b_r)))


# Test general rules for dense operators
# ======================================

# Addition
@test_throws bases.IncompatibleBases op1 + dagger(op2)
test_op_equal(op1 + op2, op2 + op1)
test_op_equal(op1 + 0*op2, op1)
test_op_equal(op1 + (op2 + op3), (op1 + op2) + op3)
test_op_equal(0.3*(op1 + op2), 0.3*op2 + 0.3*op1)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
test_op_equal(op1 - op2, op1 + (-op2))
test_op_equal(op1 - op2, op1 + (-1*op2))

# Test multiplication
@test_throws bases.IncompatibleBases op1*op2
test_state_equal(op1*(x1 + x2), op1*x1 + op1*x2)
test_state_equal((op1 + op2)*(x1 + x2), op1*x1 + op1*x2 + op2*x1 + op2*x2)
test_op_equal((op1 + op2)*dagger(op3), op1*dagger(op3) + op2*dagger(op3))
test_op_equal(0.3*(op1*dagger(op2)), op1*(0.3*dagger(op2)))

# Test division
test_op_equal(op1/7, (1/7)*op1)


# Compare arithmetic of other operators to dense operators
# ========================================================

# Case 1
result = op1 + op2 - op3
result_sparse = sparse(op1) + sparse(op2) - sparse(op3)

@test typeof(result_sparse) == SparseOperator
test_op_equal(result, result_sparse)

# Case 2
result = 0.3*op1*dagger(op2)/7
result_sparse = 0.3*sparse(op1)*dagger(sparse(op2))/7

@test typeof(result_sparse) == SparseOperator
test_op_equal(result, result_sparse)
