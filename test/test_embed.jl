using Test
using QuantumOptics
using Random, SparseArrays, LinearAlgebra

@testset "embed" begin

Random.seed!(0)

# Set up operators
spinbasis = SpinBasis(1//2)

b1 = NLevelBasis(3)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

I1 = dense(identityoperator(b1))
I2 = dense(identityoperator(b2))
I3 = dense(identityoperator(b3))

b = b1 ⊗ b2 ⊗ b3

op1 = DenseOperator(b1, b1, rand(ComplexF64, length(b1), length(b1)))
op2 = DenseOperator(b2, b2, rand(ComplexF64, length(b2), length(b2)))
op3 = DenseOperator(b3, b3, rand(ComplexF64, length(b3), length(b3)))


# Test Vector{Int}, Vector{AbstractOperator}
x = embed(b, [1,2], [op1, op2])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, [1,2], [sparse(op1), sparse(op2)])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, 1, op1)
y = op1 ⊗ I2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, 2, op2)
y = I1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, 3, op3)
y = I1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(x, y))


# Test Dict(Int=>AbstractOperator)
x = embed(b, Dict(1 => sparse(op1), 2 => sparse(op2)))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict(1 => op1, 2 => op2))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, Dict([1,3] => sparse(op1⊗op3)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict([1,3] => op1⊗op3))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(x, y))

x = embed(b, Dict([3,1] => sparse(op3⊗op1)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(dense(x), y))

x = embed(b, Dict([3,1] => op3⊗op1))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_nh(x, y))

end # testset
