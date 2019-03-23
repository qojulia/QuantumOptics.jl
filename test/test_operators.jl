using Test
using QuantumOptics
using LinearAlgebra, SparseArrays, Random

mutable struct test_operators{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR}
  basis_l::BL
  basis_r::BR
  data::Matrix{ComplexF64}
  test_operators(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new{typeof(b1),typeof(b2)}(b1, b2, data) : throw(DimensionMismatch())
end

@testset "operators" begin

Random.seed!(0)

b1 = GenericBasis(3)
b2 = GenericBasis(2)
b = b1 ⊗ b2
op1 = randoperator(b1)
op2 = randoperator(b2)
op = randoperator(b, b)
op_test = test_operators(b, b, op.data)
op_test2 = test_operators(b1, b, randoperator(b1, b).data)
op_test3 = test_operators(b1 ⊗ b2, b2 ⊗ b1, randoperator(b, b).data)
ψ = randstate(b)
ρ = randoperator(b)

@test basis(op1) == b1
@test length(op1) == length(op1.data) == length(b1)^2

@test_throws ArgumentError op_test*op_test
@test_throws ArgumentError -op_test

@test_throws ArgumentError 1 + op_test
@test_throws ArgumentError op_test + 1
@test_throws ArgumentError 1 - op_test
@test_throws ArgumentError op_test - 1

@test_throws ArgumentError dagger(op_test)
@test_throws ArgumentError op_test'
@test_throws ArgumentError identityoperator(test_operators, b, b)
@test_throws ArgumentError tr(op_test)
@test_throws ArgumentError ptrace(op_test, [1])
@test_throws ArgumentError ishermitian(op_test)
@test_throws ArgumentError dense(op_test)
@test_throws ArgumentError sparse(op_test)
@test_throws ArgumentError transpose(op_test)

@test expect(1, op1, ρ) ≈ expect(embed(b, 1, op1), ρ)
@test expect(1, op1, ψ) ≈ expect(embed(b, 1, op1), ψ)
@test expect(op, [ρ, ρ]) == [expect(op, ρ) for i=1:2]
@test expect(1, op1, [ρ, ψ]) == [expect(1, op1, ρ), expect(1, op1, ψ)]

@test variance(1, op1, ρ) ≈ variance(embed(b, 1, op1), ρ)
@test variance(1, op1, ψ) ≈ variance(embed(b, 1, op1), ψ)
@test variance(op, [ρ, ρ]) == [variance(op, ρ) for i=1:2]
@test variance(1, op1, [ρ, ψ]) == [variance(1, op1, ρ), variance(1, op1, ψ)]

@test tensor(op_test) === op_test
@test_throws ArgumentError tensor(op_test, op_test)
@test_throws ArgumentError permutesystems(op_test, [1, 2])

@test embed(b, b, [1,2], op) == embed(b, [1,2], op)
@test embed(b, Dict{Vector{Int}, SparseOperator}()) == identityoperator(b)
@test_throws bases.IncompatibleBases embed(b1⊗b2, [2], [op1])

b_comp = b⊗b
@test embed(b_comp, [1,[3,4]], [op1,op]) == dense(op1 ⊗ one(b2) ⊗ op)
@test embed(b_comp, [[1,2],4], [op,op2]) == dense(op ⊗ one(b1) ⊗ op2)
@test_throws bases.IncompatibleBases embed(b_comp, [[1,2],3], [op,op2])
@test_throws bases.IncompatibleBases embed(b_comp, [[1,3],4], [op,op2])

b_comp = b_comp⊗b_comp
OP_test1 = dense(tensor([op1,one(b2),op,one(b1),one(b2),op1,one(b2)]...))
OP_test2 = embed(b_comp, [1,[3,4],7], [op1,op,op1])
OP_dif_data = (OP_test1 - OP_test2).data
@test real(sum(abs.(OP_dif_data))) < 1e-10

b8 = b2⊗b2⊗b2
cnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
op_cnot = DenseOperator(b2⊗b2, cnot)
OP_cnot = embed(b8, [1,3], op_cnot)
@assert ptrace(OP_cnot, [2])/2. == op_cnot

@test_throws ErrorException QuantumOptics.operators.gemm!()
@test_throws ErrorException QuantumOptics.operators.gemv!()

@test_throws ArgumentError exp(sparse(op1))

@test one(b1).data == Diagonal(ones(b1.shape[1]))
@test one(op1).data == Diagonal(ones(b1.shape[1]))

@test_throws ArgumentError conj!(op_test)

end # testset
