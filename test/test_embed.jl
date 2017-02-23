using Base.Test
using QuantumOptics

@testset "embed" begin

srand(0)

# Set up operators
spinbasis = SpinBasis(1//2)

b1 = NLevelBasis(3)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

I1 = full(identityoperator(b1))
I2 = full(identityoperator(b2))
I3 = full(identityoperator(b3))

b = b1 ⊗ b2 ⊗ b3

op1 = DenseOperator(b1, b1, rand(Complex128, length(b1), length(b1)))
op2 = DenseOperator(b2, b2, rand(Complex128, length(b2), length(b2)))
op3 = DenseOperator(b3, b3, rand(Complex128, length(b3), length(b3)))


# Test Vector{Int}, Vector{Operator}
x = embed(b, [1,2], [op1, op2])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(x, y))

x = embed(b, [1,2], [sparse(op1), sparse(op2)])
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(full(x), y))

x = embed(b, 1, op1)
y = op1 ⊗ I2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(x, y))

x = embed(b, 2, op2)
y = I1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(x, y))

x = embed(b, 3, op3)
y = I1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_general(x, y))


# Test Dict(Int=>Operator)
x = embed(b, Dict(1 => sparse(op1), 2 => sparse(op2)))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(full(x), y))

x = embed(b, Dict(1 => op1, 2 => op2))
y = op1 ⊗ op2 ⊗ I3
@test 0 ≈ abs(tracedistance_general(x, y))

x = embed(b, Dict([1,3] => sparse(op1⊗op3)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_general(full(x), y))

x = embed(b, Dict([1,3] => op1⊗op3))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_general(x, y))

x = embed(b, Dict([3,1] => sparse(op3⊗op1)))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_general(full(x), y))

x = embed(b, Dict([3,1] => op3⊗op1))
y = op1 ⊗ I2 ⊗ op3
@test 0 ≈ abs(tracedistance_general(x, y))

end # testset
