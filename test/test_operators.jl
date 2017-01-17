using Base.Test
using QuantumOptics

fockbasis = FockBasis(40)
spinbasis = SpinBasis(1//2)

alpha = 0.5
beta = 1.5

a = full(destroy(fockbasis))
at = full(create(fockbasis))
n = full(number(fockbasis))

sx = full(sigmax(spinbasis))
sy = full(sigmay(spinbasis))
sz = full(sigmaz(spinbasis))
sp = full(sigmap(spinbasis))
sm = full(sigmam(spinbasis))

xket = coherentstate(fockbasis, alpha)
xbra = dagger(xket)
yket = coherentstate(fockbasis, beta)
ybra = dagger(yket)

op1 = DenseOperator(spinbasis, GenericBasis([3]), [1 1 1; 1 1 1])
op2 = DenseOperator(GenericBasis([3]), spinbasis, [1 1; 1 1; 1 1])
I = dense_identityoperator(fockbasis)


# Test creation
@test_throws DimensionMismatch DenseOperator(spinbasis, [1 1 1; 1 1 1])
@test_throws DimensionMismatch DenseOperator(spinbasis, FockBasis(3), [1 1; 1 1; 1 1])
@test_approx_eq 0. maximum(abs((dagger(op1)-op2).data))

# Test addition
@test_approx_eq 0. tracedistance(sx, sp + sm)
@test_throws bases.IncompatibleBases op1+op2

# Test substraction
@test_approx_eq 0. tracedistance(sy, -1im*(sp - sm))
@test_throws bases.IncompatibleBases op1-op2

# Test multiplication
@test_approx_eq 0. norm(I*xket - xket)
@test_approx_eq 0. norm(xbra*I - xbra)
@test_approx_eq alpha norm(a*xket)
@test_approx_eq alpha norm(xbra*at)
@test_approx_eq_eps 0. tracedistance(n, at*a) 1e-13
@test_approx_eq_eps 0. norm((5. * at)*xket - 5 * (at*xket)) 1e-13
@test_approx_eq_eps 0. norm((at * 5.)*xket - (at*xket) * 5) 1e-13
@test_throws bases.IncompatibleBases a*op1

# Test division
@test_approx_eq_eps 0. norm((at/5.)*xket - (at*xket)/5) 1e-13

# Test projector
@test_approx_eq_eps 0. norm(projector(xket)*xket - xket) 1e-13
@test_approx_eq_eps 0. norm(xbra*projector(xket) - xbra) 1e-13
@test_approx_eq_eps 0. norm(projector(xbra)*xket - xket) 1e-13
@test_approx_eq_eps 0. norm(xbra*projector(xbra) - xbra) 1e-13
@test_approx_eq_eps 0. norm(ybra*projector(yket, xbra) - xbra) 1e-13
@test_approx_eq_eps 0. norm(projector(yket, xbra)*xket - yket) 1e-13

# Test trace and normalize
op = DenseOperator(GenericBasis([3]), [1 3 2;5 2 2;-1 2 5])
@test_approx_eq 8. trace(op)
op_normalized = normalize(op)
@test_approx_eq 8. trace(op)
@test_approx_eq 1. trace(op_normalized)
op_ = normalize!(op)
@test op_ === op
@test_approx_eq 1. trace(op)

# Test operator exponential
b = GenericBasis([3])
op = DenseOperator(GenericBasis([3]), [1 3 2;5 2 2;-1 2 5])
op = op + dagger(op)
op /= norm(op.data)
v = SubspaceBasis(b, eigenstates_hermitian(op))
P = projector(v, b)
op_diag = P*op*dagger(P)
op_diag_exp = DenseOperator(v, diagm(exp(diag(op_diag.data))))
@test_approx_eq_eps 0. tracedistance(expm(op), dagger(P)*op_diag_exp*P) 1e-13

# Test identity function
@test full(I) == dense_identityoperator(a)

# Test gemv implementation
result_ket = deepcopy(xket)
operators.gemv!(complex(1.0), at, xket, complex(0.), result_ket)
@test_approx_eq 0. norm(result_ket-at*xket)

result_bra = deepcopy(xbra)
operators.gemv!(complex(1.0), xbra, at, complex(0.), result_bra)
@test_approx_eq 0. norm(result_bra-xbra*at)

# Test permutating systems
b1a = NLevelBasis(2)
b1b = SpinBasis(3//2)
b2a = SpinBasis(1//2)
b2b = FockBasis(7)
b3a = FockBasis(2)
b3b = NLevelBasis(4)

srand(0)
rho1 = DenseOperator(b1a, b1b, rand(Complex128, length(b1a), length(b1b)))
rho2 = DenseOperator(b2a, b2b, rand(Complex128, length(b2a), length(b2b)))
rho3 = DenseOperator(b3a, b3b, rand(Complex128, length(b3a), length(b3b)))

@test_approx_eq_eps 0. tracedistance_general(permutesystems(rho1⊗rho2⊗rho3, [2, 1, 3]), rho2⊗rho1⊗rho3) 1e-5
