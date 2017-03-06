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
@test tracenorm(0*rho) == 0.
@test_throws ArgumentError tracenorm(sparse(rho))

# tracenorm_general
@test tracenorm_general(0*rho) == 0.
@test_throws ArgumentError tracenorm_general(sparse(rho))

# tracedistance
@test tracedistance(rho, sigma) == 1.
@test tracedistance(rho, rho) == 0.
@test tracedistance(sigma, sigma) == 0.
@test_throws ArgumentError tracedistance(sparse(rho), sparse(rho))

# tracedistance_general
@test 1e-6 > abs(tracedistance_general(rho, sigma)) - 1.
@test tracedistance_general(rho, rho) == 0.
@test tracedistance_general(sigma, sigma) == 0.

rho = spinup(b1) ⊗ dagger(coherentstate(b2, 0.1))
@test_throws AssertionError tracedistance(rho, rho)
@test tracedistance_general(rho, rho) == 0.
@test_throws ArgumentError tracedistance_general(sparse(rho), sparse(rho))

# entropy
rho_mix = full(identityoperator(b1))/2.
@test entropy_vn(rho_mix)/log(2) == 1.0

# fidelity
rho = tensor(psi1, dagger(psi1))
@test fidelity(rho, rho) ≈ 1
@test fidelity(rho, sigma) == 0

end
