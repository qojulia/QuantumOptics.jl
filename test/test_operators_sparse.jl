using Test
using QuantumOptics
using Random, SparseArrays, LinearAlgebra


# Custom operator type for testing error msg
mutable struct TestOperator{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}; end


@testset "operators-sparse" begin

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)
sprandop(b1, b2) = sparse(randoperator(b1, b2))
sprandop(b) = sprandop(b, b)

b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)

b_l = b1a⊗b2a⊗b3a
b_r = b1b⊗b2b⊗b3b

# Test creation
@test_throws DimensionMismatch DenseOperator(b1a, spzeros(ComplexF64, 3, 2))
@test_throws DimensionMismatch DenseOperator(b1a, b1b, spzeros(ComplexF64, 3, 2))
op1 = SparseOperator(b1a, b1b, sparse([1 1 1; 1 1 1]))
op2 = sparse(DenseOperator(b1b, b1a, [1 1; 1 1; 1 1]))
@test op1 == dagger(op2)

# Test copy
op1 = sparse(randoperator(b1a))
op2 = copy(op1)
@test !(op1.data === op2.data)
op2.data[1,1] = complex(10.)
@test op1.data[1,1] != op2.data[1,1]

# Arithmetic operations
# =====================
op_zero = SparseOperator(b_l, b_r)
op1 = sprandop(b_l, b_r)
op2 = sprandop(b_l, b_r)
op3 = sprandop(b_l, b_r)
op1_ = dense(op1)
op2_ = dense(op2)
op3_ = dense(op3)

x1 = randstate(b_r)
x2 = randstate(b_r)

xbra1 = dagger(randstate(b_l))
xbra2 = dagger(randstate(b_l))

# Addition
@test_throws bases.IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1+op2, op1_+op2_)
@test 1e-14 > D(op1+op2, op1+op2_)
@test 1e-14 > D(op1+op2, op1_+op2)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1-op2, op1_-op2_)
@test 1e-14 > D(op1-op2, op1-op2_)
@test 1e-14 > D(op1-op2, op1_-op2)
@test 1e-14 > D(op1+(-op2), op1_ - op2_)
@test 1e-14 > D(op1+(-1*op2), op1_ - op2_)

# Test multiplication
@test_throws bases.IncompatibleBases op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1_*(x1 + 0.3*x2))
@test 1e-11 > D(op1*x1 + 0.3*op1*x2, op1_*x1 + 0.3*op1_*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), (op1_+op2_)*(x1+0.3*x2))

@test 1e-11 > D((xbra1 + 0.3*xbra2)*op1, (xbra1 + 0.3*xbra2)*op1_)
@test 1e-11 > D(xbra1*op1 + 0.3*xbra2*op1, xbra1*op1_ + 0.3*xbra2*op1_)
@test 1e-11 > D((xbra1+0.3*xbra2)*(op1+op2), (xbra1+0.3*xbra2)*(op1_+op2_))

@test 1e-12 > D(op1*dagger(0.3*op2), op1_*dagger(0.3*op2_))
@test 1e-12 > D(0.3*dagger(op2*dagger(op1)), 0.3*dagger(op2_*dagger(op1_)))
@test 1e-12 > D((op1 + op2)*dagger(0.3*op3), (op1_ + op2_)*dagger(0.3*op3_))
@test 1e-12 > D(0.3*op1*dagger(op3) + 0.3*op2*dagger(op3), 0.3*op1_*dagger(op3_) + 0.3*op2_*dagger(op3_))

# Test division
@test 1e-14 > D(op1/7, op1_/7)

# Conjugation
tmp = copy(op1)
conj!(tmp)
@test tmp == conj(op1) && conj(tmp.data) == op1.data

# Test identityoperator
Idense = identityoperator(DenseOperator, b_r)
I = identityoperator(SparseOperator, b_r)
@test isa(I, SparseOperator)
@test dense(I) == Idense
@test 1e-11 > D(I*x1, x1)
@test I == identityoperator(SparseOperator, b1b) ⊗ identityoperator(SparseOperator, b2b) ⊗ identityoperator(SparseOperator, b3b)

Idense = identityoperator(DenseOperator, b_l)
I = identityoperator(SparseOperator, b_l)
@test isa(I, SparseOperator)
@test dense(I) == Idense
@test 1e-11 > D(xbra1*I, xbra1)
@test I == identityoperator(SparseOperator, b1a) ⊗ identityoperator(SparseOperator, b2a) ⊗ identityoperator(SparseOperator, b3a)


# Test tr and normalize
op = sparse(DenseOperator(GenericBasis(3), [1 3 2;5 2 2;-1 2 5]))
@test 8 == tr(op)
op_normalized = normalize(op)
@test 8 == tr(op)
@test 1 == tr(op_normalized)
# op_ = normalize!(op)
# @test op_ === op
# @test 1 == tr(op)

# Test partial tr
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
b_l = b1 ⊗ b2 ⊗ b3
op1 = sprandop(b1)
op2 = sprandop(b2)
op3 = sprandop(b3)
op123 = op1 ⊗ op2 ⊗ op3
op123_ = dense(op123)

@test 1e-14 > D(ptrace(op123_, 3), ptrace(op123, 3))
@test 1e-14 > D(ptrace(op123_, 2), ptrace(op123, 2))
@test 1e-14 > D(ptrace(op123_, 1), ptrace(op123, 1))

@test 1e-14 > D(ptrace(op123_, [2,3]), ptrace(op123, [2,3]))
@test 1e-14 > D(ptrace(op123_, [1,3]), ptrace(op123, [1,3]))
@test 1e-14 > D(ptrace(op123_, [1,2]), ptrace(op123, [1,2]))

@test_throws ArgumentError ptrace(op123, [1,2,3])

# Test expect
state = randstate(b_l)
@test expect(op123, state) ≈ expect(op123_, state)

state = randoperator(b_l)
@test expect(op123, state) ≈ expect(op123_, state)


# Tensor product
# ==============
b1a = GenericBasis(2)
b1b = GenericBasis(3)
b2a = GenericBasis(1)
b2b = GenericBasis(4)
b3a = GenericBasis(1)
b3b = GenericBasis(5)
b_l = b1a ⊗ b2a ⊗ b3a
b_r = b1b ⊗ b2b ⊗ b3b
op1a = sprandop(b1a, b1b)
op1b = sprandop(b1a, b1b)
op2a = sprandop(b2a, b2b)
op2b = sprandop(b2a, b2b)
op3a = sprandop(b3a, b3b)
op1a_ = dense(op1a)
op1b_ = dense(op1b)
op2a_ = dense(op2a)
op2b_ = dense(op2b)
op3a_ = dense(op3a)
op123 = op1a ⊗ op2a ⊗ op3a
op123_ = op1a_ ⊗ op2a_ ⊗ op3a_
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test 1e-13 > D((op1a ⊗ op2a) ⊗ op3a, (op1a_ ⊗ op2a_) ⊗ op3a_)
@test 1e-13 > D(op1a ⊗ (op2a ⊗ op3a), op1a_ ⊗ (op2a_ ⊗ op3a_))
@test 1e-13 > D(op1a ⊗ (op2a ⊗ op3a), op1a_ ⊗ (op2a_ ⊗ op3a))

# Linearity
@test 1e-13 > D(op1a ⊗ (0.3*op2a), op1a_ ⊗ (0.3*op2a_))
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a_ ⊗ op2a_))
@test 1e-13 > D((0.3*op1a) ⊗ op2a, (0.3*op1a_) ⊗ op2a_)
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a_ ⊗ op2a_))
@test 1e-13 > D(0.3*(op1a ⊗ op2a), 0.3*(op1a ⊗ op2a_))

# Distributivity
@test 1e-13 > D(op1a ⊗ (op2a + op2b), op1a_ ⊗ (op2a_ + op2b_))
@test 1e-13 > D(op1a ⊗ op2a + op1a ⊗ op2b, op1a_ ⊗ op2a_ + op1a_ ⊗ op2b_)
@test 1e-13 > D((op2a + op2b) ⊗ op3a, (op2a_ + op2b_) ⊗ op3a_)
@test 1e-13 > D(op2a ⊗ op3a + op2b ⊗ op3a, op2a_ ⊗ op3a_ + op2b_ ⊗ op3a_)
@test 1e-13 > D(op2a ⊗ op3a + op2b ⊗ op3a, op2a ⊗ op3a_ + op2b_ ⊗ op3a_)

# Mixed-product property
@test 1e-13 > D((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a_ ⊗ op2a_) * dagger(op1b_ ⊗ op2b_))
@test 1e-13 > D((op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)), (op1a_*dagger(op1b_)) ⊗ (op2a_*dagger(op2b_)))
@test 1e-13 > D((op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)), (op1a_*dagger(op1b)) ⊗ (op2a_*dagger(op2b_)))

# Transpose
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a_ ⊗ op2a_))
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a ⊗ op2a_))
@test 1e-13 > D(dagger(op1a) ⊗ dagger(op2a), dagger(op1a_) ⊗ dagger(op2a_))


# Permute systems
op1 = sprandop(b1a, b1b)
op2 = sprandop(b2a, b2b)
op3 = sprandop(b3a, b3b)
op123 = op1⊗op2⊗op3

op132 = op1⊗op3⊗op2
@test 1e-14 > D(permutesystems(op123, [1, 3, 2]), op132)

op213 = op2⊗op1⊗op3
@test 1e-14 > D(permutesystems(op123, [2, 1, 3]), op213)

op231 = op2⊗op3⊗op1
@test 1e-14 > D(permutesystems(op123, [2, 3, 1]), op231)

op312 = op3⊗op1⊗op2
@test 1e-14 > D(permutesystems(op123, [3, 1, 2]), op312)

op321 = op3⊗op2⊗op1
@test 1e-14 > D(permutesystems(op123, [3, 2, 1]), op321)

# Test diagonaloperator
b = GenericBasis(4)
I = identityoperator(b)

@test diagonaloperator(b, [1, 1, 1, 1]) == I
@test diagonaloperator(b, [1., 1., 1., 1.]) == I
@test diagonaloperator(b, [1im, 1im, 1im, 1im]) == 1im*I
@test diagonaloperator(b, [0:3;]) == sparse(DenseOperator(b, Diagonal([0:3;])))

# Test gemv
op = sprandop(b_l, b_r)
op_ = dense(op)
xket = randstate(b_l)
xbra = dagger(xket)

state = randstate(b_r)
result_ = randstate(b_l)
result = deepcopy(result_)
operators.gemv!(complex(1.0), op, state, complex(0.), result)
@test 1e-13 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, op, state, beta, result)
@test 1e-13 > D(result, alpha*op_*state + beta*result_)

state = dagger(randstate(b_l))
result_ = dagger(randstate(b_r))
result = deepcopy(result_)
operators.gemv!(complex(1.0), state, op, complex(0.), result)
@test 1e-13 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, state, op, beta, result)
@test 1e-13 > D(result, alpha*state*op_ + beta*result_)

# Test gemm small version
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)

op = sprandop(b1, b2)
op_ = dense(op)

state = randoperator(b2, b3)
result_ = randoperator(b1, b3)
result = deepcopy(result_)
operators.gemm!(complex(1.), op, state, complex(0.), result)
@test 1e-12 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op, state, beta, result)
@test 1e-12 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b3, b1)
result_ = randoperator(b3, b2)
result = deepcopy(result_)
operators.gemm!(complex(1.), state, op, complex(0.), result)
@test 1e-12 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, state, op, beta, result)
@test 1e-12 > D(result, alpha*state*op_ + beta*result_)

# Test gemm big version
b1 = GenericBasis(50)
b2 = GenericBasis(60)
b3 = GenericBasis(55)

op = sprandop(b1, b2)
op_ = dense(op)

state = randoperator(b2, b3)
result_ = randoperator(b1, b3)
result = deepcopy(result_)
operators.gemm!(complex(1.), op, state, complex(0.), result)
@test 1e-11 > D(result, op_*state)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op, state, beta, result)
@test 1e-11 > D(result, alpha*op_*state + beta*result_)

state = randoperator(b3, b1)
result_ = randoperator(b3, b2)
result = deepcopy(result_)
operators.gemm!(complex(1.), state, op, complex(0.), result)
@test 1e-11 > D(result, state*op_)

result = deepcopy(result_)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, state, op, beta, result)
@test 1e-11 > D(result, alpha*state*op_ + beta*result_)

# Test remaining uncovered code
@test_throws DimensionMismatch SparseOperator(b1, b2, zeros(10, 10))
dat = sprandop(b1, b1).data
@test SparseOperator(b1, dat) == SparseOperator(b1, Matrix{ComplexF64}(dat))

@test_throws ArgumentError sparse(TestOperator{Basis,Basis}())

@test 2*SparseOperator(b1, dat) == SparseOperator(b1, dat)*2
@test copy(op1) == deepcopy(op1)

# Test Hermitian
bspin = SpinBasis(1//2)
bnlevel = NLevelBasis(2)
@test ishermitian(SparseOperator(bspin, bspin, sparse([1.0 im; -im 2.0]))) == true
@test ishermitian(SparseOperator(bspin, bnlevel, sparse([1.0 im; -im 2.0]))) == false

end # testset
