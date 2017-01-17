using Base.Test
using QuantumOptics

shape1 = [5]
shape2 = [2, 3]
shape3 = [6]

b1 = GenericBasis(shape1)
b2 = GenericBasis(shape2)
b3 = GenericBasis(shape3)

@test b1.shape == shape1
@test b2.shape == shape2
@test b1 != b2
@test b1 == b1

comp_b1 = tensor(b1, b2)
comp_b2 = tensor(b1, b1, b2)
@test comp_b1.shape == [prod(shape1), prod(shape2)]
@test comp_b2.shape == [prod(shape1), prod(shape1), prod(shape2)]

comp_b1_b2 = tensor(comp_b1, comp_b2)
@test comp_b1_b2.shape == [prod(shape1), prod(shape2), prod(shape1), prod(shape1), prod(shape2)]
@test comp_b1_b2 == CompositeBasis(b1, b2, b1, b1, b2)

@test comp_b2.shape == tensor(b1, comp_b1).shape
@test comp_b2 == tensor(b1, comp_b1)

@test ptrace(comp_b2, [1]) == ptrace(comp_b2, [2]) == comp_b1
@test ptrace(comp_b2, [1, 2]) == ptrace(comp_b1, [1])
@test ptrace(comp_b2, [2, 3]) == ptrace(comp_b1, [2])

comp1 = tensor(b1, b2, b3)
comp2 = tensor(b2, b1, b3)
@test permutesystems(comp1, [2,1,3]) == comp2
