using Base.Test
using QuantumOptics

@testset "fock" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
randstate(b) = normalize(Ket(b, rand(Complex128, length(b))))
randop(bl, br) = DenseOperator(bl, br, rand(Complex128, length(bl), length(br)))
randop(b) = randop(b, b)

basis = FockBasis(2)

# Test creation
@test basis.Nmin == 0
@test basis.Nmax == 2
@test basis.shape[1] == 3
@test_throws DimensionMismatch FockBasis(-1, 4)
@test_throws DimensionMismatch FockBasis(5, 4)


# Test equality
@test FockBasis(2) == FockBasis(2)
@test FockBasis(2) == FockBasis(0,2)
@test FockBasis(2) != FockBasis(3)
@test FockBasis(1,3) != FockBasis(2,4)


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
b1 = FockBasis(2, 5)
b2 = FockBasis(5)

@test expect(number(b1), fockstate(b1, 3)) == complex(3.)
@test expect(number(b2), fockstate(b2, 3)) == complex(3.)


# Test coherent states
b1 = FockBasis(100)
b2 = FockBasis(2, 5)
alpha = complex(3.)
a = destroy(b1)
n = number(b1)
psi = coherentstate(b1, alpha)
rho = psi ⊗ dagger(psi)

@test 1e-14 > norm(expect(a, psi) - alpha)
@test 1e-14 > norm(expect(a, rho) - alpha)
@test 1e-14 > norm(coherentstate(b1, alpha).data[3:6] - coherentstate(b2, alpha).data)
@test 1e-13 > abs(variance(n, psi) - abs(alpha)^2)
@test 1e-13 > abs(variance(n, rho) - abs(alpha)^2)

# Test qfunc
b = FockBasis(50)
alpha = complex(1., 2.)
X = [-2.1:1.5:2;]
Y = [0.13:1.4:3;]
psi = coherentstate(b, alpha)
rho = tensor(psi, dagger(psi))

Qpsi = qfunc(psi, X, Y)
Qrho = qfunc(rho, X, Y)
for (i,x)=enumerate(X), (j,y)=enumerate(Y)
    c = complex(x, y)
    q = exp(-abs2(c) - abs2(alpha) + 2*real(alpha*conj(c)))/pi
    @test 1e-14 > abs(Qpsi[i,j] - q)
    @test 1e-14 > abs(Qrho[i,j] - q)
    @test 1e-14 > abs(qfunc(psi, c) - q)
    @test 1e-14 > abs(qfunc(rho, c) - q)
end

b = FockBasis(50)
psi = randstate(b)
rho = randop(b)
X = [-2.1:1.5:2;]
Y = [0.13:1.4:3;]

Qpsi = qfunc(psi, X, Y)
Qrho = qfunc(rho, X, Y)
for (i,x)=enumerate(X), (j,y)=enumerate(Y)
    c = complex(x, y)
    state = coherentstate(b, c)
    q_rho = dagger(state) * rho * state/pi
    q_psi = abs2(dagger(state) *psi)/pi
    @test 1e-14 > abs(Qpsi[i,j] - q_psi)
    @test 1e-14 > abs(Qrho[i,j] - q_rho)
    @test 1e-14 > abs(qfunc(psi, c) - q_psi)
    @test 1e-14 > abs(qfunc(rho, c) - q_rho)
end

end # testset
