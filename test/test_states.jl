using Base.Test
using QuantumOptics

@testset "states" begin

srand(0)

D(x1::Number, x2::Number) = abs(x2-x1)
D(x1::StateVector, x2::StateVector) = norm(x2-x1)
randstate(b) = Ket(b, rand(Complex128, length(b)))

b1 = GenericBasis(3)
b2 = GenericBasis(5)
b = b1 ⊗ b2

bra = Bra(b)
ket = Ket(b)

# Test creation
@test_throws DimensionMismatch Bra(b, [1, 2])
@test_throws DimensionMismatch Ket(b, [1, 2])
@test 0 ≈ norm(bra)
@test 0 ≈ norm(ket)
@test_throws bases.IncompatibleBases bra*Ket(b1)

# Arithmetic operations
# =====================
bra_b1 = dagger(randstate(b1))
bra_b2 = dagger(randstate(b2))

ket_b1 = randstate(b1)
ket_b2 = randstate(b2)

# Addition
@test_throws bases.IncompatibleBases bra_b1 + bra_b2
@test_throws bases.IncompatibleBases ket_b1 + ket_b2
@test 1e-14 > D(bra_b1 + Bra(b1), bra_b1)
@test 1e-14 > D(ket_b1 + Ket(b1), ket_b1)
@test 1e-14 > D(bra_b1 + dagger(ket_b1), dagger(ket_b1) + bra_b1)

# Subtraction
@test_throws bases.IncompatibleBases bra_b1 - bra_b2
@test_throws bases.IncompatibleBases ket_b1 - ket_b2
@test 1e-14 > D(bra_b1 - Bra(b1), bra_b1)
@test 1e-14 > D(ket_b1 - Ket(b1), ket_b1)
@test 1e-14 > D(bra_b1 - dagger(ket_b1), -dagger(ket_b1) + bra_b1)

# Multiplication
@test 1e-14 > D(-3*ket_b1, 3*(-ket_b1))
@test 1e-14 > D(0.3*(bra_b1 - dagger(ket_b1)), 0.3*bra_b1 - dagger(0.3*ket_b1))
@test 1e-14 > D(0.3*(bra_b1 - dagger(ket_b1)), bra_b1*0.3 - dagger(ket_b1*0.3))
@test 0 ≈ bra*ket
@test 1e-14 > D((bra_b1 ⊗ bra_b2)*(ket_b1 ⊗ ket_b2), (bra_b1*ket_b1)*(bra_b2*ket_b2))

# Norm
basis = FockBasis(0, 1)
bra = Bra(basis, [3im, -4])
ket = Ket(basis, [-4im, 3])
@test 5 ≈ norm(bra)
@test 5 ≈ norm(ket)

bra_normalized = normalize(bra)
ket_normalized = normalize(ket)
@test 5 ≈ norm(bra)
@test 5 ≈ norm(ket)
@test 1 ≈ norm(bra_normalized)
@test 1 ≈ norm(ket_normalized)

bra_copy = deepcopy(bra)
ket_copy = deepcopy(ket)
normalize!(bra_copy)
normalize!(ket_copy)
@test 5 ≈ norm(bra)
@test 5 ≈ norm(ket)
@test 1 ≈ norm(bra_copy)
@test 1 ≈ norm(ket_copy)

# Test basis state
b1 = GenericBasis(2)
b2 = GenericBasis(3)
b = b1 ⊗ b2
x1 = basisstate(b1, 2)
x2 = basisstate(b2, 1)

@test norm(x1) == 1
@test x1.data[2] == 1
@test basisstate(b, [2, 1]) == x1 ⊗ x2

# Partial Trace
basis = FockBasis(0, 1)
a = normalize(Bra(basis, [1im, 0]))
b = normalize(Bra(basis, [1, 2]))
c = normalize(Bra(CompositeBasis(basis, basis), [1im, 2im, 0, 0]))
@test 0 ≈ norm(tensor(a, b) - c)
@test 0 ≈ tracedistance(operators.ptrace(c, 1), tensor(dagger(b), b))
@test 1e-16 > tracedistance(operators.ptrace(c, 2), tensor(dagger(a), a))


# Test permutating systems
b1 = NLevelBasis(2)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

srand(0)
psi1 = normalize(Ket(b1, rand(Complex128, length(b1))))
psi2 = normalize(Ket(b2, rand(Complex128, length(b2))))
psi3 = normalize(Ket(b3, rand(Complex128, length(b3))))

psi123 = psi1 ⊗ psi2 ⊗ psi3
psi213 = psi2 ⊗ psi1 ⊗ psi3

c = dagger(psi213)*permutesystems(psi123, [2,1,3])

@test 1e-5 > abs(1.-c)

end # testset
