using Test
using QuantumOptics
using LinearAlgebra, Random


@testset "operators-lazysum" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws AssertionError LazySum()
@test_throws AssertionError LazySum([1., 2.], [randoperator(b_l)])
@test_throws AssertionError LazySum(randoperator(b_l, b_r), sparse(randoperator(b_l, b_l)))
@test_throws AssertionError LazySum(randoperator(b_l, b_r), sparse(randoperator(b_r, b_r)))

# Test copy
op1 = 2*LazySum(randoperator(b_l, b_r), sparse(randoperator(b_l, b_r)))
op2 = copy(op1)
@test op1 == op2
@test !(op1 === op2)
op2.operators[1].data[1,1] = complex(10.)
@test op1.operators[1].data[1,1] != op2.operators[1].data[1,1]
op2.factors[1] = 3.
@test op2.factors[1] != op1.factors[1]

# Test dense & sparse
op1 = randoperator(b_l, b_r)
op2 = sparse(randoperator(b_l, b_r))
@test 0.1*op1 + 0.3*dense(op2) == dense(LazySum([0.1, 0.3], [op1, op2]))
@test 0.1*sparse(op1) + 0.3*op2 == sparse(LazySum([0.1, 0.3], [op1, op2]))


# Arithmetic operations
# =====================
op1a = randoperator(b_l, b_r)
op1b = randoperator(b_l, b_r)
op2a = randoperator(b_l, b_r)
op2b = randoperator(b_l, b_r)
op3a = randoperator(b_l, b_r)
op1 = LazySum([0.1, 0.3], [op1a, sparse(op1b)])
op1_ = 0.1*op1a + 0.3*op1b
op2 = LazySum([0.7, 0.9], [sparse(op2a), op2b])
op2_ = 0.7*op2a + 0.9*op2b
op3 = LazySum(op3a)
op3_ = op3a

x1 = Ket(b_r, rand(ComplexF64, length(b_r)))
x2 = Ket(b_r, rand(ComplexF64, length(b_r)))
xbra1 = Bra(b_l, rand(ComplexF64, length(b_l)))

# Addition
@test_throws bases.IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1+op2, op1_+op2_)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1 - op2, op1_ - op2_)
@test 1e-14 > D(op1 + (-op2), op1_ - op2_)
@test 1e-14 > D(op1 + (-1*op2), op1_ - op2_)

# Test multiplication
@test_throws ArgumentError op1*op2
@test LazySum([0.1, 0.1], [op1a, op2a]) == LazySum(op1a, op2a)*0.1
@test LazySum([0.1, 0.1], [op1a, op2a]) == 0.1*LazySum(op1a, op2a)
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), (op1_+op2_)*(x1+0.3*x2))
@test 1e-12 > D(dagger(x1)*dagger(0.3*op2), dagger(x1)*dagger(0.3*op2_))

# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Test identityoperator
Idense = identityoperator(DenseOperator, b_r)
I = identityoperator(LazySum, b_r)
@test isa(I, LazySum)
@test dense(I) == Idense
@test 1e-11 > D(I*x1, x1)

Idense = identityoperator(DenseOperator, b_l)
I = identityoperator(LazySum, b_l)
@test isa(I, LazySum)
@test dense(I) == Idense
@test 1e-11 > D(xbra1*I, xbra1)

# Test tr and normalize
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op3 = randoperator(b_l)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test tr(op_) ≈ tr(op)
op_normalized = normalize(op)
@test tr(op_) ≈ tr(op)
@test 1 ≈ tr(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test tr(op) != tr(op_copy)
@test 1 ≈ tr(op_copy)

# Test partial tr
op1 = randoperator(b_l)
op2 = randoperator(b_l)
op3 = randoperator(b_l)
op123 = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op123_ = 0.1*op1 + 0.3*op2 + 1.2*op3

@test 1e-14 > D(ptrace(op123_, 3), ptrace(op123, 3))
@test 1e-14 > D(ptrace(op123_, 2), ptrace(op123, 2))
@test 1e-14 > D(ptrace(op123_, 1), ptrace(op123, 1))

@test 1e-14 > D(ptrace(op123_, [2,3]), ptrace(op123, [2,3]))
@test 1e-14 > D(ptrace(op123_, [1,3]), ptrace(op123, [1,3]))
@test 1e-14 > D(ptrace(op123_, [1,2]), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])

# Test expect
state = Ket(b_l, rand(ComplexF64, length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

state = DenseOperator(b_l, b_l, rand(ComplexF64, length(b_l), length(b_l)))
@test expect(op123, state) ≈ expect(op123_, state)

# Permute systems
op1a = randoperator(b1a)
op2a = randoperator(b2a)
op3a = randoperator(b3a)
op1b = randoperator(b1a)
op2b = randoperator(b2a)
op3b = randoperator(b3a)
op1c = randoperator(b1a)
op2c = randoperator(b2a)
op3c = randoperator(b3a)
op123a = op1a⊗op2a⊗op3a
op123b = op1b⊗op2b⊗op3b
op123c = op1c⊗op2c⊗op3c
op = LazySum([0.3, 0.7, 1.2], [op123a, sparse(op123b), op123c])
op_ = 0.3*op123a + 0.7*op123b + 1.2*op123c

@test 1e-14 > D(permutesystems(op, [1, 3, 2]), permutesystems(op_, [1, 3, 2]))
@test 1e-14 > D(permutesystems(op, [2, 1, 3]), permutesystems(op_, [2, 1, 3]))
@test 1e-14 > D(permutesystems(op, [2, 3, 1]), permutesystems(op_, [2, 3, 1]))
@test 1e-14 > D(permutesystems(op, [3, 1, 2]), permutesystems(op_, [3, 1, 2]))
@test 1e-14 > D(permutesystems(op, [3, 2, 1]), permutesystems(op_, [3, 2, 1]))


# Test gemv
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

state = Ket(b_r, rand(ComplexF64, length(b_r)))
result_ = Ket(b_l, rand(ComplexF64, length(b_l)))
result = deepcopy(result_)
operators.gemv!(complex(1.), op, state, complex(0.), result)
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, op, state, beta, result)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = Bra(b_l, rand(ComplexF64, length(b_l)))
result_ = Bra(b_r, rand(ComplexF64, length(b_r)))
result = deepcopy(result_)
operators.gemv!(complex(1.), state, op, complex(0.), result)
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, state, op, beta, result)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)
op = LazySum([0.1, 0.3, 1.2], [op1, op2, op3])
op_ = 0.1*op1 + 0.3*op2 + 1.2*op3

state = randoperator(b_r, b_r)
result_ = randoperator(b_l, b_r)
result = deepcopy(result_)
operators.gemm!(complex(1.), op, state, complex(0.), result)
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op, state, beta, result)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b_l, b_l)
result_ = randoperator(b_l, b_r)
result = deepcopy(result_)
operators.gemm!(complex(1.), state, op, complex(0.), result)
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, state, op, beta, result)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

end # testset
