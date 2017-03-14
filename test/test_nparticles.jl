using Base.Test
using QuantumOptics

@testset "nparticles" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

Bosons = BosonicNParticleBasis
Fermions = FermionicNParticleBasis

# Test creation
Nparticles = 2
Nmodes = 3
b_bosons = Bosons(Nmodes, Nparticles)
@test length(b_bosons.occupations) == 6
@test_throws ArgumentError Bosons(4, 2, Vector{Int}[[-1, 0, 2, 1]])
@test_throws ArgumentError Bosons(4, 2, Vector{Int}[[1, 0, 0, 1, 0]])
@test_throws ArgumentError Bosons(4, 2, Vector{Int}[[1, 0, 0, 2]])

b_fermions = FermionicNParticleBasis(Nmodes, Nparticles)
@test length(b_fermions.occupations) == 3
@test_throws ArgumentError Fermions(4, 2, Vector{Int}[[-1, 1, 1, 1]])
@test_throws ArgumentError Fermions(4, 2, Vector{Int}[[1, 0, 0, 1, 0]])
@test_throws ArgumentError Fermions(4, 2, Vector{Int}[[0, 0, 0, 2]])


# Test equality
@test b_bosons != Bosons(Nmodes, Nparticles, Vector{Int}[[2, 0, 0]])
@test b_bosons == Bosons(GenericBasis(Nmodes), Nparticles)

@test b_fermions != Fermions(Nmodes, Nparticles, Vector{Int}[[1, 0, 1]])
@test b_fermions == Fermions(GenericBasis(Nmodes), Nparticles)


# Test nparticleoperator
# ======================
# Calculate single particle operator in second quantization
b_single = GenericBasis(Nmodes)
x = randoperator(b_single)
y = randoperator(b_single)

# Bosons
X = nparticleoperator(b_bosons, x)
Y = nparticleoperator(b_bosons, y)

@test 1e-14 > D(X + Y, nparticleoperator(b_bosons, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-14 > D(X, nparticleoperator(b_bosons, x_))
@test 1e-14 > D(Y, nparticleoperator(b_bosons, y_))
@test 1e-14 > D(X + Y, nparticleoperator(b_bosons, x_ + y_))

# Fermions
X = nparticleoperator(b_fermions, x)
Y = nparticleoperator(b_fermions, y)

@test 1e-14 > D(X + Y, nparticleoperator(b_fermions, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-14 > D(X, nparticleoperator(b_fermions, x_))
@test 1e-14 > D(Y, nparticleoperator(b_fermions, y_))
@test 1e-14 > D(X + Y, nparticleoperator(b_fermions, x_ + y_))


# Calculate particle-particle interaction operator in second quantization
x = randoperator(b_single ⊗ b_single)
y = randoperator(b_single ⊗ b_single)

# Bosons
X = nparticleoperator(b_bosons, x)
Y = nparticleoperator(b_bosons, y)

@test 1e-14 > D(X + Y, nparticleoperator(b_bosons, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-14 > D(X, nparticleoperator(b_bosons, x_))
@test 1e-14 > D(Y, nparticleoperator(b_bosons, y_))
@test 1e-14 > D(X + Y, nparticleoperator(b_bosons, x_ + y_))

# Fermions
X = nparticleoperator(b_fermions, x)
Y = nparticleoperator(b_fermions, y)

@test 1e-14 > D(X + Y, nparticleoperator(b_fermions, x + y))

x_ = sparse(x)
y_ = sparse(y)
@test 1e-14 > D(X, nparticleoperator(b_fermions, x_))
@test 1e-14 > D(Y, nparticleoperator(b_fermions, y_))
@test 1e-14 > D(X + Y, nparticleoperator(b_fermions, x_ + y_))

# Test expect_firstquantization
# =============================
x = randoperator(b_single)
y = randoperator(b_single)
X = nparticleoperator(b_bosons, x)
Y = nparticleoperator(b_bosons, y)

psi = randstate(b_bosons)

@test expect_firstquantization(x, psi) ≈ expect(X, psi⊗dagger(psi))
@test expect_firstquantization(x, Y) ≈ expect(X, Y)
@test expect_firstquantization(sparse(x), psi) ≈ expect(X, psi⊗dagger(psi))
@test expect_firstquantization(sparse(x), Y) ≈ expect(X, Y)

end # testset
