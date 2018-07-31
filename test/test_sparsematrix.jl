using Test
using QuantumOptics.sparsematrix

# SparseMatrix = quantumoptics.sparsematrix.SparseMatrix
const SparseMatrix = SparseMatrixCSC{ComplexF64, Int}


@testset "sparsematrix" begin

# Set up test matrices
A = rand(ComplexF64, 5, 5)
A_sp = sparse(A)

B = eye(ComplexF64, 5)
B_sp = speye(ComplexF64, 5)

C = rand(ComplexF64, 3, 3)
C[2,:] = 0
C_sp = sparse(C)

R_sp = A_sp + B_sp
R = A + B


# Test arithmetic
@test 0 ≈ norm(dense(R_sp) - R)
@test 0 ≈ norm(dense(ComplexF64(0.5,0)*A_sp) - 0.5*A)
@test 0 ≈ norm(dense(A_sp/2) - A/2)
@test 0 ≈ norm(dense(A_sp*B_sp) - A*B)

# Test kronecker product
@test 0 ≈ norm(dense(kron(A_sp, C_sp)) - kron(A, C))
@test 0 ≈ norm(dense(kron(A_sp, B_sp)) - kron(A, B))

end # testset
