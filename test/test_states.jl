using Test
using QuantumOptics
using LinearAlgebra, Random

@testset "states" begin

Random.seed!(0)

D(x1::Number, x2::Number) = abs(x2-x1)
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(7)
b = b1 ⊗ b2

bra = Bra(b)
ket = Ket(b)

# Test creation
@test_throws DimensionMismatch Bra(b, [1, 2])
@test_throws DimensionMismatch Ket(b, [1, 2])
@test 0 ≈ norm(bra)
@test 0 ≈ norm(ket)
@test_throws bases.IncompatibleBases bra*Ket(b1)
@test bra == bra
@test length(bra) == length(bra.data) == 15
@test length(ket) == length(ket.data) == 15
@test bases.basis(ket) == b
@test bases.basis(bra) == b
@test bra != basisstate(b, 1)

# Test copy
psi1 = randstate(b1)
psi2 = copy(psi1)
@test psi1.data == psi2.data
@test !(psi1.data === psi2.data)
psi2.data[1] = complex(10.)
@test psi1.data[1] != psi2.data[1]

# Arithmetic operations
# =====================
bra_b1 = dagger(randstate(b1))
bra_b2 = dagger(randstate(b2))
bra_b3 = randstate(b3)'

ket_b1 = randstate(b1)
ket_b2 = randstate(b2)
ket_b3 = randstate(b3)

# Addition
@test_throws DimensionMismatch bra_b1 + bra_b2
@test_throws DimensionMismatch ket_b1 + ket_b2
@test 1e-14 > D(bra_b1 + Bra(b1), bra_b1)
@test 1e-14 > D(ket_b1 + Ket(b1), ket_b1)
@test 1e-14 > D(bra_b1 + dagger(ket_b1), dagger(ket_b1) + bra_b1)

# Subtraction
@test_throws DimensionMismatch bra_b1 - bra_b2
@test_throws DimensionMismatch ket_b1 - ket_b2
@test 1e-14 > D(bra_b1 - Bra(b1), bra_b1)
@test 1e-14 > D(ket_b1 - Ket(b1), ket_b1)
@test 1e-14 > D(bra_b1 - dagger(ket_b1), -dagger(ket_b1) + bra_b1)

# Multiplication
@test 1e-14 > D(-3*ket_b1, 3*(-ket_b1))
@test 1e-14 > D(0.3*(bra_b1 - dagger(ket_b1)), 0.3*bra_b1 - dagger(0.3*ket_b1))
@test 1e-14 > D(0.3*(bra_b1 - dagger(ket_b1)), bra_b1*0.3 - dagger(ket_b1*0.3))
@test 0 ≈ bra*ket
@test 1e-14 > D((bra_b1 ⊗ bra_b2)*(ket_b1 ⊗ ket_b2), (bra_b1*ket_b1)*(bra_b2*ket_b2))

# Tensor product
@test tensor(ket_b1) == ket_b1
@test 1e-14 > D((ket_b1 ⊗ ket_b2) ⊗ ket_b3, ket_b1 ⊗ (ket_b2 ⊗ ket_b3))
@test 1e-14 > D((bra_b1 ⊗ bra_b2) ⊗ bra_b3, bra_b1 ⊗ (bra_b2 ⊗ bra_b3))

ket_b1b2 = ket_b1 ⊗ ket_b2
shape = (ket_b1b2.basis.shape...,)
idx = LinearIndices(shape)[2, 3]
@test ket_b1b2.data[idx] == ket_b1.data[2]*ket_b2.data[3]
ket_b1b2b3 = ket_b1 ⊗ ket_b2 ⊗ ket_b3
@test ket_b1b2b3 == tensor(ket_b1, ket_b2, ket_b3)
shape = (ket_b1b2b3.basis.shape...,)
idx = LinearIndices(shape)[1, 4, 3]
@test ket_b1b2b3.data[idx] == ket_b1.data[1]*ket_b2.data[4]*ket_b3.data[3]


# Norm
bf = FockBasis(1)
bra = Bra(bf, [3im, -4])
ket = Ket(bf, [-4im, 3])
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

# Test permutating systems
b1 = GenericBasis(2)
b2 = GenericBasis(5)
b3 = FockBasis(3)

psi1 = randstate(b1)
psi2 = randstate(b2)
psi3 = randstate(b3)

psi123 = psi1 ⊗ psi2 ⊗ psi3
psi132 = psi1 ⊗ psi3 ⊗ psi2
psi213 = psi2 ⊗ psi1 ⊗ psi3
psi231 = psi2 ⊗ psi3 ⊗ psi1
psi312 = psi3 ⊗ psi1 ⊗ psi2
psi321 = psi3 ⊗ psi2 ⊗ psi1

@test 1e-14 > D(psi132, permutesystems(psi123, [1, 3, 2]))
@test 1e-14 > D(psi213, permutesystems(psi123, [2, 1, 3]))
@test 1e-14 > D(psi231, permutesystems(psi123, [2, 3, 1]))
@test 1e-14 > D(psi312, permutesystems(psi123, [3, 1, 2]))
@test 1e-14 > D(psi321, permutesystems(psi123, [3, 2, 1]))

@test 1e-14 > D(dagger(psi132), permutesystems(dagger(psi123), [1, 3, 2]))
@test 1e-14 > D(dagger(psi213), permutesystems(dagger(psi123), [2, 1, 3]))
@test 1e-14 > D(dagger(psi231), permutesystems(dagger(psi123), [2, 3, 1]))
@test 1e-14 > D(dagger(psi312), permutesystems(dagger(psi123), [3, 1, 2]))
@test 1e-14 > D(dagger(psi321), permutesystems(dagger(psi123), [3, 2, 1]))

# Test Broadcasting
@test_throws bases.IncompatibleBases psi123 .= psi132
@test_throws bases.IncompatibleBases psi123 .+ psi132
bra123 = dagger(psi123)
bra132 = dagger(psi132)
@test_throws ArgumentError psi123 .+ bra123
@test_throws bases.IncompatibleBases bra123 .= bra132
@test_throws bases.IncompatibleBases bra123 .+ bra132
psi_ = copy(psi123)
psi_ .+= psi123
@test psi_ == 2*psi123
bra_ = copy(bra123)
bra_ .= 3*bra123
@test bra_ == 3*dagger(psi123)

end # testset
