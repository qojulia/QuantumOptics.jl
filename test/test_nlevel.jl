using Base.Test
using QuantumOptics

energies = [0., 1.1, 2.6]
b = NLevelBasis(energies)

@test b == NLevelBasis(energies)
@test b != NLevelBasis([0.1, 1.1, 2.6])
@test full(transition(b, 2, 1)) == basis_ket(b, 2) ⊗ basis_bra(b, 1)