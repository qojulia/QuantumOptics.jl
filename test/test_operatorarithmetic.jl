using Base.Test
using QuantumOptics

@testset "operatorarithmetic" begin

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

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
@test 1e-14 > D(op1 + op2, op2 + op1)
@test 1e-14 > D(op1 + 0*op2, op1)
@test 1e-14 > D(op1 + (op2 + op3), (op1 + op2) + op3)
@test 1e-14 > D(0.3*(op1 + op2), 0.3*op2 + 0.3*op1)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1 - op2, op1 + (-op2))
@test 1e-14 > D(op1 - op2, op1 + (-1*op2))

# Test multiplication
@test_throws bases.IncompatibleBases op1*op2
test_state_equal(op1*(x1 + x2), op1*x1 + op1*x2)
test_state_equal((op1 + op2)*(x1 + x2), op1*x1 + op1*x2 + op2*x1 + op2*x2)
@test 1e-12 > D((op1 + op2)*dagger(op3), op1*dagger(op3) + op2*dagger(op3))
@test 1e-12 > D(0.3*(op1*dagger(op2)), op1*(0.3*dagger(op2)))

# Test division
@test 1e-14 > D(op1/7, (1/7)*op1)


# Compare arithmetic of other operators to dense operators
# ========================================================

# Case 1
result = 0.1*op1 + 0.3*op2 - op3/4
result_sparse = 0.1*sparse(op1) + 0.3*sparse(op2) - sparse(op3)/4
result_lsum = 0.1*lazy(op1) + 0.3*lazy(op2) - lazy(op3)/4

@test isa(result_sparse, SparseOperator)
@test isa(result_lsum, LazySum)

@test 1e-14 > D(result, result_sparse)
@test 1e-14 > D(result, result_lsum)

# Case 2
result = 0.3*op1*dagger(op2)/7
result_sparse = 0.3*sparse(op1)*dagger(sparse(op2))/7
result_lprod = 0.3*lazy(op1)*dagger(lazy(op2))/7

@test isa(result_sparse, SparseOperator)
@test isa(result_lprod, LazyProduct)

@test 1e-13 > D(result, result_sparse)
@test 1e-13 > D(result, result_lprod)

end # testset
