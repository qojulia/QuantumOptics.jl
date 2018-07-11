using Base.Test
using QuantumOptics


mutable struct test_operators <: Operator
  basis_l::Basis
  basis_r::Basis
  data::Matrix{Complex128}
  test_operators(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

@testset "operators" begin

srand(0)

b1 = GenericBasis(5)
b2 = GenericBasis(3)
b = b1 ⊗ b2
op1 = randoperator(b1)
op = randoperator(b, b)
op_test = test_operators(b, b, op.data)
op_test2 = test_operators(b1, b, randoperator(b1, b).data)
op_test3 = test_operators(b1 ⊗ b2, b2 ⊗ b1, randoperator(b, b).data)
ψ = randstate(b)
ρ = randoperator(b)

@test basis(op1) == b1
@test length(op1) == length(op1.data) == 25

@test_throws ArgumentError op_test*op_test
@test_throws ArgumentError -op_test

@test_throws ArgumentError 1 + op_test
@test_throws ArgumentError op_test + 1
@test_throws ArgumentError 1 - op_test
@test_throws ArgumentError op_test - 1

@test_throws ArgumentError dagger(op_test)
@test_throws ArgumentError identityoperator(test_operators, b, b)
@test_throws ArgumentError trace(op_test)
@test_throws ArgumentError ptrace(op_test, [1])
@test_throws ArgumentError ishermitian(op_test)
@test_throws ArgumentError full(op_test)
@test_throws ArgumentError sparse(op_test)

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

@test embed(b, b, 1, op) == embed(b, 1, op)
@test embed(b, Dict{Vector{Int}, SparseOperator}()) == identityoperator(b)

@test_throws ErrorException QuantumOptics.operators.gemm!()
@test_throws ErrorException QuantumOptics.operators.gemv!()

@test_throws ArgumentError expm(sparse(op1))

@test one(b1).data == diagm(ones(b1.shape[1]))
@test one(op1).data == diagm(ones(b1.shape[1]))

@test_throws ArgumentError conj!(op_test)

end # testset
