using LinearAlgebra
using Test

using QuantumOptics

@testset "pauli" begin

@test_throws MethodError PauliBasis(1.4)

# Test conversion of unitary matrices to superoperators.
q2 = PauliBasis(2)
CZ = DenseOperator(q2, q2, diagm(0 => [1,1,1,-1]))
CZ_sop = SuperOperator(CZ)

# Test conversion of unitary matrices to superoperators.
@test diag(CZ_sop.data) ==  ComplexF64[1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1]
@test CZ_sop.basis_l == CZ_sop.basis_r == (q2, q2)

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = PauliTransferMatrix(CZ_sop)

CZ_ptm_test = zeros(Float64, (16, 16))
CZ_ptm_test[[1,30,47,52,72,91,117,140,166,185,205,210,227,256]] = ones(14)
CZ_ptm_test[[106,151]] = -1 * ones(2)

@test CZ_ptm.data == CZ_ptm_test
@test CZ_ptm == PauliTransferMatrix(ChiMatrix(CZ))

# Test conversion among all three bases.
cphase = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(1im*.6)]

q2 = PauliBasis(2)

CPHASE = DenseOperator(q2, cphase)

CPHASE_sop = SuperOperator(CPHASE)
CPHASE_chi = ChiMatrix(CPHASE)
CPHASE_ptm = PauliTransferMatrix(CPHASE)

@test ChiMatrix(CPHASE_sop) == CPHASE_chi
@test ChiMatrix(CPHASE_ptm) == CPHASE_chi
@test SuperOperator(CPHASE_chi) == CPHASE_sop
@test SuperOperator(CPHASE_ptm) == CPHASE_sop
@test PauliTransferMatrix(CPHASE_sop) == CPHASE_ptm
@test PauliTransferMatrix(CPHASE_chi) == CPHASE_ptm

end # testset
