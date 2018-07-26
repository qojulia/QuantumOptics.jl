using Test
using QuantumOptics.sparsematrix

# SparseMatrix = quantumoptics.sparsematrix.SparseMatrix
const SparseMatrix = SparseMatrixCSC{ComplexF64, Int}


@testset "sparsematrix" begin

# Set up test matrices
A = rand(Complex128, 5, 5)
A_sp = sparse(A)

B = eye(Complex128, 5)
B_sp = speye(Complex128, 5)

C = rand(Complex128, 3, 3)
C[2,:] = 0
C_sp = sparse(C)

R_sp = A_sp + B_sp
R = A + B


# Test arithmetic
@test 0 ≈ norm(full(R_sp) - R)
@test 0 ≈ norm(full(Complex128(0.5,0)*A_sp) - 0.5*A)
@test 0 ≈ norm(full(A_sp/2) - A/2)
@test 0 ≈ norm(full(A_sp*B_sp) - A*B)

# Test kronecker product
@test 0 ≈ norm(full(kron(A_sp, C_sp)) - kron(A, C))
@test 0 ≈ norm(full(kron(A_sp, B_sp)) - kron(A, B))

end # testset
