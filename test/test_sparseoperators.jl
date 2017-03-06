using Base.Test
using QuantumOptics

type TestOperator <: Operator
end

@testset "sparseoperator" begin

srand(0)

# Set up operators
spinbasis = SpinBasis(1//2)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)

sx_dense = full(sx)
sy_dense = full(sy)
sz_dense = full(sz)

@test isa(sx_dense, DenseOperator)
@test isa(sparse(sx_dense), SparseOperator)
@test sparse(sx_dense) == sx

b = FockBasis(3) ⊗ SpinBasis(1//2)
op = SparseOperator(b, b, sparse(rand(Complex128, length(b), length(b))))
state = Ket(b, rand(Complex128, length(b)))
@test expect(op, state) ≈ dagger(state)*op*state
@test expect(op, state) ≈ dagger(state)*full(op)*state
state = DenseOperator(b, b, rand(Complex128, length(b), length(b)))
@test expect(op, state) ≈ trace(op*state)
@test expect(op, state) ≈ trace(full(op)*state)

b = FockBasis(3)
I = identityoperator(b)

a = TestOperator()

@test_throws ArgumentError sparse(a)

@test diagonaloperator(b, [1, 1, 1, 1]) == I
@test diagonaloperator(b, [1., 1., 1., 1.]) == I
@test diagonaloperator(b, [1im, 1im, 1im, 1im]) == 1im*I
@test diagonaloperator(b, [0:3;]) == number(b)

end # testset
