using Base.Test
using QuantumOptics


type test_operators <: Operator
  basis_l::Basis
  basis_r::Basis
  data::Matrix{Complex128}
  test_operators(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

@testset "operators" begin

srand(0)

b = GenericBasis(5)
b_comp = b ⊗ b
op_dense = randoperator(b, b)
op_test = test_operators(b, b, op_dense.data)
ψ = randstate(b)
ρ = ψ ⊗ dagger(ψ)

@test_throws ArgumentError op_test*op_test
@test_throws ArgumentError -op_test

@test_throws ArgumentError 1 + op_test
@test_throws ArgumentError op_test + 1
@test_throws ArgumentError 1 - op_test
@test_throws ArgumentError op_test - 1

@test_throws ArgumentError dagger(op_test)
@test_throws ArgumentError identityoperator(test_operators, b, b)
@test_throws ArgumentError trace(op_test)
@test_throws ArgumentError ptrace(op_test, [1, 2])

@test expect(op_dense, [ρ, ρ]) == [expect(op_dense, ρ) for i=1:2]

@test_throws ArgumentError tensor(op_test, op_test)
@test_throws ArgumentError permutesystems(op_test, [1, 2])

@test embed(b_comp, b_comp, 1, op_dense) == embed(b_comp, 1, op_dense)
@test embed(b_comp, Dict{Vector{Int}, SparseOperator}()) == identityoperator(b_comp)

@test_throws ErrorException QuantumOptics.operators.gemm!()
@test_throws ErrorException QuantumOptics.operators.gemv!()

end # testset
