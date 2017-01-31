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

op1a = DenseOperator(b1a, b1b, rand(Complex128, length(b1a), length(b1b)))
op1b = DenseOperator(b1a, b1b, rand(Complex128, length(b1a), length(b1b)))
op2a = DenseOperator(b2a, b2b, rand(Complex128, length(b2a), length(b2b)))
op2b = DenseOperator(b2a, b2b, rand(Complex128, length(b2a), length(b2b)))
op3a = DenseOperator(b3a, b3b, rand(Complex128, length(b3a), length(b3b)))
op3b = DenseOperator(b3a, b3b, rand(Complex128, length(b3a), length(b3b)))


# Test general rules for dense operators
# ======================================

# Correct bases
op123 = op1a ⊗ op2a ⊗ op3a
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
test_op_equal((op1a ⊗ op2a) ⊗ op3a, op1a ⊗ (op2a ⊗ op3a))

# Linearity
test_op_equal(op1a ⊗ (0.3*op2a), 0.3*(op1a ⊗ op2a))
test_op_equal((0.3*op1a) ⊗ op2a, 0.3*(op1a ⊗ op2a))

# Distributivity
test_op_equal(op1a ⊗ (op2a + op2b), op1a ⊗ op2a + op1a ⊗ op2b)
test_op_equal((op2a + op2b) ⊗ op3a, op2a ⊗ op3a + op2b ⊗ op3a)

# Mixed-product property
test_op_equal((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)))

# Transpose
test_op_equal(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))


# Compare tensor product of other operators to dense operators
# ============================================================

# Case 1
op = op1a ⊗ op2a
op_sparse = sparse(op1a) ⊗ sparse(op2a)

@test typeof(op_sparse) == SparseOperator
test_op_equal(op_sparse, op)

# Case 2
op = op1a ⊗ (op2a * dagger(op2b)) ⊗ (op3a + op3b)
op_sparse = sparse(op1a) ⊗ (sparse(op2a) * dagger(sparse(op2b))) ⊗ (sparse(op3a) + sparse(op3b))

@test typeof(op_sparse) == SparseOperator
test_op_equal(op_sparse, op)
