using Test
using QuantumOptics

@testset "basis" begin

shape1 = [5]
shape2 = [2, 3]
shape3 = [6]

b1 = GenericBasis(shape1)
b2 = GenericBasis(shape2)
b3 = GenericBasis(shape3)

@test b1.shape == shape1
@test b2.shape == shape2
@test b1 != b2
@test b1 != FockBasis(2)
@test b1 == b1

@test tensor(b1) == b1
comp_b1 = tensor(b1, b2)
comp_uni = b1 ⊗ b2
comp_b2 = tensor(b1, b1, b2)
@test comp_b1.shape == [prod(shape1), prod(shape2)]
@test comp_uni.shape == [prod(shape1), prod(shape2)]
@test comp_b2.shape == [prod(shape1), prod(shape1), prod(shape2)]

@test b1^3 == CompositeBasis(b1, b1, b1)
@test (b1⊗b2)^2 == CompositeBasis(b1, b2, b1, b2)
@test_throws ArgumentError b1^(0)

comp_b1_b2 = tensor(comp_b1, comp_b2)
@test comp_b1_b2.shape == [prod(shape1), prod(shape2), prod(shape1), prod(shape1), prod(shape2)]
@test comp_b1_b2 == CompositeBasis(b1, b2, b1, b1, b2)

@test_throws ArgumentError tensor()
@test comp_b2.shape == tensor(b1, comp_b1).shape
@test comp_b2 == tensor(b1, comp_b1)

@test_throws ArgumentError ptrace(comp_b1, [1, 2])
@test ptrace(comp_b2, [1]) == ptrace(comp_b2, [2]) == comp_b1 == ptrace(comp_b2, 1)
@test ptrace(comp_b2, [1, 2]) == ptrace(comp_b1, [1])
@test ptrace(comp_b2, [2, 3]) == ptrace(comp_b1, [2])

comp1 = tensor(b1, b2, b3)
comp2 = tensor(b2, b1, b3)
@test permutesystems(comp1, [2,1,3]) == comp2

@test !QuantumOptics.bases.equal_bases([b1, b2], [b1, b3])
@test !QuantumOptics.bases.multiplicable(comp1, b1 ⊗ b2 ⊗ NLevelBasis(prod(b3.shape)))

end # testset
