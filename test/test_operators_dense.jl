using Base.Test
using QuantumOptics

@testset "operators-dense" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
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
@test_throws DimensionMismatch DenseOperator(b1a, [1 1 1; 1 1 1])
@test_throws DimensionMismatch DenseOperator(b1a, b1b, [1 1; 1 1; 1 1])
op1 = DenseOperator(b1a, b1b, [1 1 1; 1 1 1])
op2 = DenseOperator(b1b, b1a, [1 1; 1 1; 1 1])
@test op1 == dagger(op2)

# Test copy
op1 = randoperator(b1a)
op2 = copy(op1)
@test !(op1.data === op2.data)
op2.data[1,1] = complex(10.)
@test op1.data[1,1] != op2.data[1,1]


# Arithmetic operations
# =====================
op_zero = DenseOperator(b_l, b_r)
op1 = randoperator(b_l, b_r)
op2 = randoperator(b_l, b_r)
op3 = randoperator(b_l, b_r)

x1 = Ket(b_r, rand(Complex128, length(b_r)))
x2 = Ket(b_r, rand(Complex128, length(b_r)))

xbra1 = Bra(b_l, rand(Complex128, length(b_l)))
xbra2 = Bra(b_l, rand(Complex128, length(b_l)))

# Addition
@test_throws bases.IncompatibleBases op1 + dagger(op2)
@test 1e-14 > D(op1 + op_zero, op1)
@test 1e-14 > D(op1 + op2, op2 + op1)
@test 1e-14 > D(op1 + (op2 + op3), (op1 + op2) + op3)

# Subtraction
@test_throws bases.IncompatibleBases op1 - dagger(op2)
@test 1e-14 > D(op1-op_zero, op1)
@test 1e-14 > D(op1-op2, op1 + (-op2))
@test 1e-14 > D(op1-op2, op1 + (-1*op2))
@test 1e-14 > D(op1-op2-op3, op1-(op2+op3))

# Test multiplication
@test_throws bases.IncompatibleBases op1*op2
@test 1e-11 > D(op1*(x1 + 0.3*x2), op1*x1 + 0.3*op1*x2)
@test 1e-11 > D((op1+op2)*(x1+0.3*x2), op1*x1 + 0.3*op1*x2 + op2*x1 + 0.3*op2*x2)

@test 1e-11 > D((xbra1+0.3*xbra2)*op1, xbra1*op1 + 0.3*xbra2*op1)
@test 1e-11 > D((xbra1+0.3*xbra2)*(op1+op2), xbra1*op1 + 0.3*xbra2*op1 + xbra1*op2 + 0.3*xbra2*op2)

@test 1e-12 > D(op1*dagger(0.3*op2), 0.3*dagger(op2*dagger(op1)))
@test 1e-12 > D((op1 + op2)*dagger(0.3*op3), 0.3*op1*dagger(op3) + 0.3*op2*dagger(op3))
@test 1e-12 > D(0.3*(op1*dagger(op2)), op1*(0.3*dagger(op2)))

# Test division
@test 1e-14 > D(op1/7, (1/7)*op1)

# Test identityoperator
I = identityoperator(DenseOperator, b_r)
@test isa(I, DenseOperator)
@test identityoperator(SparseOperator, b_r) == sparse(I)
@test 1e-11 > D(I*x1, x1)
@test I == identityoperator(DenseOperator, b1b) ⊗ identityoperator(DenseOperator, b2b) ⊗ identityoperator(DenseOperator, b3b)

I = identityoperator(DenseOperator, b_l)
@test isa(I, DenseOperator)
@test identityoperator(SparseOperator, b_l) == sparse(I)
@test 1e-11 > D(xbra1*I, xbra1)
@test I == identityoperator(DenseOperator, b1a) ⊗ identityoperator(DenseOperator, b2a) ⊗ identityoperator(DenseOperator, b3a)

# Test trace and normalize
op = DenseOperator(GenericBasis(3), [1 3 2;5 2 2;-1 2 5])
@test 8 == trace(op)
op_normalized = normalize(op)
@test 8 == trace(op)
@test 1 == trace(op_normalized)
op_copy = deepcopy(op)
normalize!(op_copy)
@test trace(op) != trace(op_copy)
@test 1 ≈ trace(op_copy)

# Test partial trace of state vectors
psi1 = 0.1*randstate(b1a)
psi2 = 0.3*randstate(b2a)
psi3 = 0.7*randstate(b3a)
psi12 = psi1 ⊗ psi2
psi13 = psi1 ⊗ psi3
psi23 = psi2 ⊗ psi3
psi123 = psi1 ⊗ psi2 ⊗ psi3

@test 1e-14 > D(0.1^2*0.3^2*psi3 ⊗ dagger(psi3), ptrace(psi123, [1, 2]))
@test 1e-14 > D(0.1^2*0.7^2*psi2 ⊗ dagger(psi2), ptrace(psi123, [1, 3]))
@test 1e-14 > D(0.3^2*0.7^2*psi1 ⊗ dagger(psi1), ptrace(psi123, [2, 3]))

@test 1e-14 > D(0.1^2*psi23 ⊗ dagger(psi23), ptrace(psi123, 1))
@test 1e-14 > D(0.3^2*psi13 ⊗ dagger(psi13), ptrace(psi123, 2))
@test 1e-14 > D(0.7^2*psi12 ⊗ dagger(psi12), ptrace(psi123, 3))

@test 1e-14 > D(ptrace(psi123, [1, 2]), dagger(ptrace(dagger(psi123), [1, 2])))
@test 1e-14 > D(ptrace(psi123, 3), dagger(ptrace(dagger(psi123), 3)))

# Test partial trace of operators
op1 = randoperator(b1a)
op2 = randoperator(b2a)
op3 = randoperator(b3a)
op123 = op1 ⊗ op2 ⊗ op3

@test 1e-14 > D(op1⊗op2*trace(op3), ptrace(op123, 3))
@test 1e-14 > D(op1⊗op3*trace(op2), ptrace(op123, 2))
@test 1e-14 > D(op2⊗op3*trace(op1), ptrace(op123, 1))

@test 1e-14 > D(op1*trace(op2)*trace(op3), ptrace(op123, [2,3]))
@test 1e-14 > D(op2*trace(op1)*trace(op3), ptrace(op123, [1,3]))
@test 1e-14 > D(op3*trace(op1)*trace(op2), ptrace(op123, [1,2]))

@test 1e-14 > abs(trace(op1)*trace(op2)*trace(op3) - ptrace(op123, [1,2,3]))

# Test expect
state = Ket(b_l, rand(Complex128, length(b_l)))
@test expect(op123, state) ≈ dagger(state)*op123*state

state = DenseOperator(b_l, b_l, rand(Complex128, length(b_l), length(b_l)))
@test expect(op123, state) ≈ trace(op123*state)


# Tensor product
# ==============
op1a = randoperator(b1a, b1b)
op1b = randoperator(b1a, b1b)
op2a = randoperator(b2a, b2b)
op2b = randoperator(b2a, b2b)
op3a = randoperator(b3a, b3b)
op123 = op1a ⊗ op2a ⊗ op3a
@test op123.basis_l == b_l
@test op123.basis_r == b_r

# Associativity
@test 1e-13 > D((op1a ⊗ op2a) ⊗ op3a, op1a ⊗ (op2a ⊗ op3a))

# Linearity
@test 1e-13 > D(op1a ⊗ (0.3*op2a), 0.3*(op1a ⊗ op2a))
@test 1e-13 > D((0.3*op1a) ⊗ op2a, 0.3*(op1a ⊗ op2a))

# Distributivity
@test 1e-13 > D(op1a ⊗ (op2a + op2b), op1a ⊗ op2a + op1a ⊗ op2b)
@test 1e-13 > D((op2a + op2b) ⊗ op3a, op2a ⊗ op3a + op2b ⊗ op3a)

# Mixed-product property
@test 1e-13 > D((op1a ⊗ op2a) * dagger(op1b ⊗ op2b), (op1a*dagger(op1b)) ⊗ (op2a*dagger(op2b)))

# Transpose
@test 1e-13 > D(dagger(op1a ⊗ op2a), dagger(op1a) ⊗ dagger(op2a))


# Permute systems
op1 = randoperator(b1a)
op2 = randoperator(b2a)
op3 = randoperator(b3a)
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


# Test projector
xket = normalize(Ket(b_l, rand(Complex128, length(b_l))))
yket = normalize(Ket(b_l, rand(Complex128, length(b_l))))
xbra = dagger(xket)
ybra = dagger(yket)

@test 1e-13 > D(projector(xket)*xket, xket)
@test 1e-13 > D(xbra*projector(xket), xbra)
@test 1e-13 > D(projector(xbra)*xket, xket)
@test 1e-13 > D(xbra*projector(xbra), xbra)
@test 1e-13 > D(ybra*projector(yket, xbra), xbra)
@test 1e-13 > D(projector(yket, xbra)*xket, yket)

# Test operator exponential
op = randoperator(b1a)
@test 1e-13 > D(op^2, op*op)
@test 1e-13 > D(op^3, op*op*op)
@test 1e-13 > D(op^4, op*op*op*op)

# Test gemv
op = randoperator(b_l)
xket = normalize(Ket(b_l, rand(Complex128, length(b_l))))
xbra = dagger(xket)

result_ket = deepcopy(xket)
operators.gemv!(complex(1.0), op, xket, complex(0.), result_ket)
@test 0 ≈ D(result_ket, op*xket)

result_ket = deepcopy(xket)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, op, xket, beta, result_ket)
@test 1e-15 > D(result_ket, alpha*op*xket + beta*xket)

result_bra = deepcopy(xbra)
operators.gemv!(complex(1.0), xbra, op, complex(0.), result_bra)
@test 0 ≈ D(result_bra, xbra*op)

result_bra = deepcopy(xbra)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemv!(alpha, xbra, op, beta, result_bra)
@test 1e-15 > D(result_bra, alpha*xbra*op + beta*xbra)

# Test gemm
op1 = randoperator(b_l)
op2 = randoperator(b_l)

result = copy(op1)
operators.gemm!(complex(1.0), op1, op2, complex(0.), result)
@test 1e-15 > D(result, op1*op2)

result = copy(op1)
alpha = complex(1.5)
beta = complex(2.1)
operators.gemm!(alpha, op1, op2, beta, result)
@test 1e-15 > D(result, alpha*op1*op2 + beta*op1)

end # testset
