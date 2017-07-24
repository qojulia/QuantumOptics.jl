using Base.Test
using QuantumOptics


@testset "operators-lazytensor" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(6)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

op1 = randoperator(b1a, b1b)
op2 = randoperator(b2a, b2b)
op3 = randoperator(b3a, b3b)

# Test creation
@test_throws AssertionError LazyTensor(b_l, b_r, [1], [randoperator(b1a)])
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], [op1])
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], [op1, sparse(randoperator(b_l, b_l))])
@test_throws AssertionError LazyTensor(b_l, b_r, [1, 2], [randoperator(b_r, b_r), sparse(op2)])

@test LazyTensor(b_l, b_r, [2, 1], [op2, op1]) == LazyTensor(b_l, b_r, [1, 2], [op1, op2])
x = randoperator(b2a)
@test LazyTensor(b_l, 2, x) == LazyTensor(b_l, b_l, [2], [x])

# Test copy
x = 2*LazyTensor(b_l, b_r, [1,2], [randoperator(b1a, b1b), sparse(randoperator(b2a, b2b))])
x_ = copy(x)
@test x == x_
@test !(x === x_)
x_.operators[1].data[1,1] = complex(10.)
@test x.operators[1].data[1,1] != x_.operators[1].data[1,1]
x_.factor = 3.
@test x_.factor != x.factor
x_.indices[2] = 100
@test x_.indices != x.indices


# Test full & sparse
I2 = identityoperator(b2a, b2b)
x = LazyTensor(b_l, b_r, [1, 3], [op1, sparse(op3)], 0.3)
@test 1e-12 > D(0.3*op1⊗full(I2)⊗op3, full(x))
@test 1e-12 > D(0.3*sparse(op1)⊗I2⊗sparse(op3), sparse(x))

# Test suboperators
@test operators_lazytensor.suboperator(x, 1) == op1
@test operators_lazytensor.suboperator(x, 3) == sparse(op3)
@test operators_lazytensor.suboperators(x, [1, 3]) == [op1, sparse(op3)]


# Arithmetic operations
# =====================
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I1 = full(identityoperator(b1a, b1b))
I2 = full(identityoperator(b2a, b2b))
I3 = full(identityoperator(b3a, b3b))
op1 = LazyTensor(b_l, b_r, [1, 3], [subop1, sparse(subop3)], 0.1)
op1_ = 0.1*subop1 ⊗ I2 ⊗ subop3
op2 = LazyTensor(b_l, b_r, [2, 3], [sparse(subop2), subop3], 0.7)
op2_ = 0.7*I1 ⊗ subop2 ⊗ subop3
op3 = 0.3*LazyTensor(b_l, b_r, 3, subop3)
op3_ = 0.3*I1 ⊗ I2 ⊗ subop3

x1 = Ket(b_r, rand(Complex128, length(b_r)))
x2 = Ket(b_r, rand(Complex128, length(b_r)))
xbra1 = Bra(b_l, rand(Complex128, length(b_l)))
xbra2 = Bra(b_l, rand(Complex128, length(b_l)))

# Addition
@test_throws ArgumentError op1 + op2
@test_throws ArgumentError op1 - op2
@test 1e-14 > D(-op1_, -op1)

# Test multiplication
@test_throws bases.IncompatibleBases op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))
@test 1e-12 > D(op1_*dagger(0.3*op2), op1_*dagger(0.3*op2_))
@test 1e-12 > D(dagger(0.3*op2)*op1_, dagger(0.3*op2_)*op1_)
@test 1e-12 > D(dagger(0.3*op2)*op1, dagger(0.3*op2_)*op1_)


# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Test identityoperator
Idense = identityoperator(DenseOperator, b_r)
I = identityoperator(LazyTensor, b_r)
@test isa(I, LazyTensor)
@test full(I) == Idense
@test 1e-11 > D(I*x1, x1)
@test I == identityoperator(LazyTensor, b1b) ⊗ identityoperator(LazyTensor, b2b) ⊗ identityoperator(LazyTensor, b3b)

Idense = identityoperator(DenseOperator, b_l)
I = identityoperator(LazyTensor, b_l)
@test isa(I, LazyTensor)
@test full(I) == Idense
@test 1e-11 > D(xbra1*I, xbra1)
@test I == identityoperator(LazyTensor, b1a) ⊗ identityoperator(LazyTensor, b2a) ⊗ identityoperator(LazyTensor, b3a)


# Test trace and normalize
subop1 = randoperator(b1a)
I2 = full(identityoperator(b2a))
subop3 = randoperator(b3a)
op = LazyTensor(b_l, b_l, [1, 3], [subop1, sparse(subop3)], 0.1)
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test trace(op) ≈ trace(op_)
op_normalized = normalize(op)
@test trace(op_) ≈ trace(op)
@test 1 ≈ trace(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test trace(op) != trace(op_copy)
@test 1 ≈ trace(op_copy)

# Test partial trace
subop1 = randoperator(b1a)
I2 = full(identityoperator(b2a))
subop3 = randoperator(b3a)
op = LazyTensor(b_l, b_l, [1, 3], [subop1, sparse(subop3)], 0.1)
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test 1e-14 > D(ptrace(op_, 3), ptrace(op, 3))
@test 1e-14 > D(ptrace(op_, 2), ptrace(op, 2))
@test 1e-14 > D(ptrace(op_, 1), ptrace(op, 1))

@test 1e-14 > D(ptrace(op_, [2,3]), ptrace(op, [2,3]))
@test 1e-14 > D(ptrace(op_, [1,3]), ptrace(op, [1,3]))
@test 1e-14 > D(ptrace(op_, [1,2]), ptrace(op, [1,2]))

@test_throws ArgumentError ptrace(op, [1,2,3])

# Test expect
state = Ket(b_l, rand(Complex128, length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

state = DenseOperator(b_l, b_l, rand(Complex128, length(b_l), length(b_l)))
@test expect(op, state) ≈ expect(op_, state)

# Permute systems
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = full(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], [subop1, sparse(subop3)])*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = full(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], [subop1, sparse(subop3)])*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

state = Ket(b_r, rand(Complex128, length(b_r)))
result_ = Ket(b_l, rand(Complex128, length(b_l)))
result = deepcopy(result_)
operators.gemv!(complex(1.), op, state, complex(0.), result)
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, op, state, beta, result)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = Bra(b_l, rand(Complex128, length(b_l)))
result_ = Bra(b_r, rand(Complex128, length(b_r)))
result = deepcopy(result_)
operators.gemv!(complex(1.), state, op, complex(0.), result)
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, state, op, beta, result)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
b_l2 = GenericBasis(17)
b_r2 = GenericBasis(13)
subop1 = randoperator(b1a, b1b)
subop2 = randoperator(b2a, b2b)
subop3 = randoperator(b3a, b3b)
I2 = full(identityoperator(b2a, b2b))
op = LazyTensor(b_l, b_r, [1, 3], [subop1, sparse(subop3)])*0.1
op_ = 0.1*subop1 ⊗ I2 ⊗ subop3

state = randoperator(b_r, b_r2)
result_ = randoperator(b_l, b_r2)
result = deepcopy(result_)
operators.gemm!(complex(1.), op, state, complex(0.), result)
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op, state, beta, result)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b_l2, b_l)
result_ = randoperator(b_l2, b_r)
result = deepcopy(result_)
operators.gemm!(complex(1.), state, op, complex(0.), result)
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, state, op, beta, result)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

end # testset
