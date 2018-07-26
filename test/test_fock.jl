using Test
using QuantumOptics

@testset "fock" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_nh(full(op1), full(op2)))
randstate(b) = normalize(Ket(b, rand(Complex128, length(b))))
randop(bl, br) = DenseOperator(bl, br, rand(Complex128, length(bl), length(br)))
randop(b) = randop(b, b)

basis = FockBasis(2)

# Test creation
@test basis.N == 2
@test basis.shape[1] == 3
@test_throws DimensionMismatch FockBasis(-1)

# Test equality
@test FockBasis(2) == FockBasis(2)
@test FockBasis(2) != FockBasis(3)

# Test operators
@test number(basis) == SparseOperator(basis, spdiagm(Complex128[0, 1, 2]))
@test destroy(basis) == SparseOperator(basis, sparse(Complex128[0 1 0; 0 0 sqrt(2); 0 0 0]))
@test create(basis) == SparseOperator(basis, sparse(Complex128[0 0 0; 1 0 0; 0 sqrt(2) 0]))
@test number(basis) == dagger(number(basis))
@test create(basis) == dagger(destroy(basis))
@test destroy(basis) == dagger(create(basis))
@test 1e-15 > D(create(basis)*destroy(basis), number(basis))

# Test application onto statevectors
@test create(basis)*fockstate(basis, 0) == fockstate(basis, 1)
@test create(basis)*fockstate(basis, 1) == sqrt(2)*fockstate(basis, 2)
@test dagger(fockstate(basis, 0))*destroy(basis) == dagger(fockstate(basis, 1))
@test dagger(fockstate(basis, 1))*destroy(basis) == sqrt(2)*dagger(fockstate(basis, 2))

@test destroy(basis)*fockstate(basis, 1) == fockstate(basis, 0)
@test destroy(basis)*fockstate(basis, 2) == sqrt(2)*fockstate(basis, 1)
@test dagger(fockstate(basis, 1))*create(basis) == dagger(fockstate(basis, 0))
@test dagger(fockstate(basis, 2))*create(basis) == sqrt(2)*dagger(fockstate(basis, 1))

# Test displacement operator
b = FockBasis(30)
alpha = complex(0.5, 0.3)
d = displace(b, alpha)
a = destroy(b)
@test 1e-12 > D(d*dagger(d), identityoperator(b))
@test 1e-12 > D(dagger(d)*d, identityoperator(b))
@test 1e-12 > D(dagger(d), displace(b, -alpha))
@test 1e-15 > norm(coherentstate(b, alpha) - displace(b, alpha)*fockstate(b, 0))

# Test Fock states
b = FockBasis(5)
@test expect(number(b), fockstate(b, 3)) == complex(3.)

# Test coherent states
b = FockBasis(100)
alpha = complex(3.)
a = destroy(b)
n = number(b)
psi = coherentstate(b, alpha)
rho = dm(psi)

@test 1e-14 > norm(expect(a, psi) - alpha)
@test 1e-14 > norm(expect(a, rho) - alpha)
@test 1e-13 > abs(variance(n, psi) - abs(alpha)^2)
@test 1e-13 > abs(variance(n, rho) - abs(alpha)^2)

end # testset
