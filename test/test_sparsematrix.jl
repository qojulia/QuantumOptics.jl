using Test
using QuantumOptics.sparsematrix
using SparseArrays, LinearAlgebra

# SparseMatrix = quantumoptics.sparsematrix.SparseMatrix
const SparseMatrix = SparseMatrixCSC{ComplexF64, Int}


@testset "sparsematrix" begin

# Set up test matrices
A = rand(ComplexF64, 5, 5)
A_sp = sparse(A)

B = Matrix{ComplexF64}(I, 5, 5)
B_sp = sparse(ComplexF64(1)*I, 5, 5)

C = rand(ComplexF64, 3, 3)
C[2,:] .= 0
C_sp = sparse(C)

R_sp = A_sp + B_sp
R = A + B


# Test arithmetic
@test 0 ≈ norm(Matrix(R_sp) - R)
@test 0 ≈ norm(Matrix(ComplexF64(0.5,0)*A_sp) - 0.5*A)
@test 0 ≈ norm(Matrix(A_sp/2) - A/2)
@test 0 ≈ norm(Matrix(A_sp*B_sp) - A*B)

# Test kronecker product
@test 0 ≈ norm(Matrix(kron(A_sp, C_sp)) - kron(A, C))
@test 0 ≈ norm(Matrix(kron(A_sp, B_sp)) - kron(A, B))

end # testset
