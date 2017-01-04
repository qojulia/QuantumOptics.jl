using Base.Test
using QuantumOptics

energies = [0., 1.1, 2.6]
N = 3
b = NLevelBasis(N)

@test b == NLevelBasis(N)
@test b != NLevelBasis(N+1)
@test full(transition(b, 2, 1)) == basis_ket(b, 2) ⊗ basis_bra(b, 1)