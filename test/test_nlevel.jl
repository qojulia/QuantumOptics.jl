using Base.Test
using QuantumOptics

@testset "nlevel" begin

N = 3
b = NLevelBasis(N)

# Test basis equality
@test b == NLevelBasis(N)
@test b != NLevelBasis(N+1)

# Test transition operator
@test_throws BoundsError transition(b, 0, 1)
@test_throws BoundsError transition(b, N+1, 1)
@test_throws BoundsError transition(b, 1, 0)
@test_throws BoundsError transition(b, 1, N+1)
@test full(transition(b, 2, 1)) == basis_ket(b, 2) ⊗ basis_bra(b, 1)

# Test nlevel states
@test_throws BoundsError nlevelstate(b, 0)
@test_throws BoundsError nlevelstate(b, N+1)
@test norm(nlevelstate(b, 1)) == 1.
@test norm(dagger(nlevelstate(b, 1))*nlevelstate(b, 2)) == 0.
@test norm(dagger(nlevelstate(b, 1))*transition(b, 1, 2)*nlevelstate(b, 2)) == 1.

end # testset
