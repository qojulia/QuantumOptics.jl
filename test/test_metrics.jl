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

# ptranspose
e = spinup(b1)
g = spindown(b1)
psi3 = (e ⊗ g - g ⊗ e)/sqrt(2)

rho = dm(psi3)
rho_pT1 = ptranspose(rho, 1)
rho_pT1_an = 0.5*(dm(e ⊗ g) + dm(g ⊗ e) - (e ⊗ e) ⊗ dagger(g ⊗ g) - (g ⊗ g) ⊗ dagger(e ⊗ e))
rho_pT2 = ptranspose(rho, 2)
@test rho_pT1.data ≈ rho_pT1_an.data
@test rho_pT2.data ≈ rho_pT1_an.data

@test_throws AssertionError ptranspose(e ⊗ dagger(psi1))
@test_throws AssertionError ptranspose(dm(e))

rho = rho ⊗ dm(g)
@test PPT(rho, 1) == PPT(rho, 2) == false
@test PPT(rho, 3)

@test negativity(rho, 1) == negativity(rho, 2) ≈ 0.5
@test isapprox(negativity(rho, 3), 0.0, atol=1e-15)

@test logarithmic_negativity(rho, 1) == logarithmic_negativity(rho, 2) ≈ 1.0
@test isapprox(logarithmic_negativity(rho, 3), 0.0, atol=1e-15)

end # testset
