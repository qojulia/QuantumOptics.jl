using LinearAlgebra
using Test

using QuantumOptics

@testset "pauli" begin

@test_throws MethodError PauliBasis(1.4)

# Test conversion of unitary matrices to superoperators.
q2 = PauliBasis(2)
CZ = DenseOperator(q2, q2, diagm(0 => [1,1,1,-1]))
sopCZ = SuperOperator(CZ)

# Test conversion of superoperator to Pauli transfer matrix.
ptmCZ = PauliTransferMatrix(sopCZ)
cz = zeros(Float64, (16, 16))
for idx in [1,30,47,52,72,91,117,140,166,185,205,210,227,256]
    cz[idx] = 1
end
for idx in [106,151]
    cz[idx] = -1
end
@test ptmCZ.data == cz

end # testset
