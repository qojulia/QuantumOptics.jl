using Base.Test
using QuantumOptics

@testset "operators" begin

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
I = full(identityoperator(fockbasis))


# Test creation
@test_throws DimensionMismatch DenseOperator(spinbasis, [1 1 1; 1 1 1])
@test_throws DimensionMismatch DenseOperator(spinbasis, FockBasis(3), [1 1; 1 1; 1 1])
@test 0 ≈ maximum(abs((dagger(op1)-op2).data))

# Test projector
@test 1e-13 > norm(projector(xket)*xket - xket)
@test 1e-13 > norm(xbra*projector(xket) - xbra)
@test 1e-13 > norm(projector(xbra)*xket - xket)
@test 1e-13 > norm(xbra*projector(xbra) - xbra)
@test 1e-13 > norm(ybra*projector(yket, xbra) - xbra)
@test 1e-13 > norm(projector(yket, xbra)*xket - yket)

# Test trace and normalize
op = DenseOperator(GenericBasis([3]), [1 3 2;5 2 2;-1 2 5])
@test 8 == trace(op)
op_normalized = normalize(op)
@test 8 == trace(op)
@test 1 == trace(op_normalized)
op_ = normalize!(op)
@test op_ === op
@test 1 == trace(op)

# Test operator exponential
b = GenericBasis([3])
op = DenseOperator(GenericBasis([3]), [1 3 2;5 2 2;-1 2 5])
op = op + dagger(op)
op /= norm(op.data)
v = SubspaceBasis(b, eigenstates_hermitian(op))
P = projector(v, b)
op_diag = P*op*dagger(P)
op_diag_exp = DenseOperator(v, diagm(exp(diag(op_diag.data))))
@test 1e-13 > tracedistance(expm(op), dagger(P)*op_diag_exp*P)

# Test gemv implementation
result_ket = deepcopy(xket)
operators.gemv!(complex(1.0), at, xket, complex(0.), result_ket)
@test 0 ≈ norm(result_ket-at*xket)

result_bra = deepcopy(xbra)
operators.gemv!(complex(1.0), xbra, at, complex(0.), result_bra)
@test 0 ≈ norm(result_bra-xbra*at)

end # testset
