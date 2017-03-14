using Base.Test
using QuantumOptics

@testset "fock" begin

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))

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
X = [-2:0.5:2;]
Y = [0:0.5:3;]
rho = tensor(coherentstate(b, alpha), dagger(coherentstate(b, alpha)))
Q = qfunc(rho, X, Y)
for (i,x)=enumerate(X), (j,y)=enumerate(Y)
    c = complex(x, y)
    @test 1e-14 > Q[i,j]-exp(-abs2(c) - abs2(alpha) + 2*real(alpha*conj(c)))/pi
end

end # testset
