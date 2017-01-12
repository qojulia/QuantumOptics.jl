using Base.Test
using QuantumOptics


# Set up operators
spinbasis = SpinBasis(1//2)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)

sx_dense = full(sx)
sy_dense = full(sy)

@test typeof(sx_dense) == DenseOperator
@test typeof(sparse(sx_dense)) == SparseOperator
@test sparse(sx_dense) == sx

b = FockBasis(3)
I = sparse_identityoperator(b)
I_dense = dense_identityoperator(b)


s = tensor(sx, sy)
s_dense = tensor(sx_dense, sy_dense)

@test typeof(I) == SparseOperator
@test typeof(I_dense) == DenseOperator
@test_approx_eq 0. norm((I_dense-full(I)).data)
@test_approx_eq 0. norm((s_dense - full(s)).data)

@test I == identityoperator(destroy(b))

type A <: Operator
end

a = A()

@test_throws ArgumentError sparse(a)


@test diagonaloperator(b, [1, 1, 1, 1]) == I
@test diagonaloperator(b, [1., 1., 1., 1.]) == I
@test diagonaloperator(b, [1im, 1im, 1im, 1im]) == 1im*I
@test diagonaloperator(b, [0:3;]) == number(b)
