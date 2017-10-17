using Base.Test
using QuantumOptics

@testset "metrics" begin

b1 = SpinBasis(1//2)
b2 = FockBasis(6)

psi1 = spinup(b1) ⊗ coherentstate(b2, 0.1)
psi2 = spindown(b1) ⊗ fockstate(b2, 2)

rho = tensor(psi1, dagger(psi1))
sigma = tensor(psi2, dagger(psi2))

# tracenorm
@test tracenorm(0*rho) ≈ 0.
@test tracenorm_h(0*rho) ≈ 0.
@test tracenorm_nh(0*rho) ≈ 0.

@test tracenorm(rho) ≈ 1.
@test tracenorm_h(rho) ≈ 1.
@test tracenorm_nh(rho) ≈ 1.

@test_throws ArgumentError tracenorm(sparse(rho))
@test_throws ArgumentError tracenorm_h(sparse(rho))
@test_throws ArgumentError tracenorm_nh(sparse(rho))

# tracedistance
@test tracedistance(rho, sigma) ≈ 1.
@test tracedistance_h(rho, sigma) ≈ 1.
@test tracedistance_nh(rho, sigma) ≈ 1.

@test tracedistance(rho, rho) ≈ 0.
@test tracedistance_h(rho, rho) ≈ 0.
@test tracedistance_nh(rho, rho) ≈ 0.

@test tracedistance(sigma, sigma) ≈ 0.
@test tracedistance_h(sigma, sigma) ≈ 0.
@test tracedistance_nh(sigma, sigma) ≈ 0.

@test_throws ArgumentError tracedistance(sparse(rho), sparse(rho))
@test_throws ArgumentError tracedistance_h(sparse(rho), sparse(rho))
@test_throws ArgumentError tracedistance_nh(sparse(rho), sparse(rho))

# tracedistance
@test tracedistance(rho, sigma) ≈ 1.
@test tracedistance(rho, rho) ≈ 0.
@test tracedistance(sigma, sigma) ≈ 0.

rho = spinup(b1) ⊗ dagger(coherentstate(b2, 0.1))
@test_throws bases.IncompatibleBases tracedistance(rho, rho)
@test_throws bases.IncompatibleBases tracedistance_h(rho, rho)

@test tracedistance_nh(rho, rho) ≈ 0.

# entropy
rho_mix = full(identityoperator(b1))/2.
@test entropy_vn(rho_mix)/log(2) ≈ 1

# fidelity
rho = tensor(psi1, dagger(psi1))
@test fidelity(rho, rho) ≈ 1
@test 1e-20 > abs2(fidelity(rho, sigma))

end
