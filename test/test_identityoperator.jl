using Base.Test
using QuantumOptics

b1 = NLevelBasis(2)
b2 = SpinBasis(3//2)
b3 = FockBasis(2)
b = b1⊗b2⊗b3

function test_op_equal(op1, op2)
    @test_approx_eq_eps 0. tracedistance_general(full(op1), full(op2)) 1e-11
end

Idense = identityoperator(DenseOperator, b)
Isparse = identityoperator(SparseOperator, b)

@test typeof(Idense) == DenseOperator
@test typeof(Isparse) == SparseOperator

@test full(Isparse) == Idense
@test Isparse == identityoperator(b)

@test Isparse == identityoperator(b1) ⊗ identityoperator(b2) ⊗ identityoperator(b3)
@test Idense == identityoperator(DenseOperator, b1) ⊗ identityoperator(DenseOperator, b2) ⊗ identityoperator(DenseOperator, b3)


op = DenseOperator(b, rand(Complex128, length(b), length(b)))

test_op_equal(op, identityoperator(op)*op)
test_op_equal(op, op*identityoperator(op))

test_op_equal(sparse(op), identityoperator(op)*op)
test_op_equal(sparse(op), sparse(op)*identityoperator(sparse(op)))
