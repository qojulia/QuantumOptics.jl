using LinearAlgebra
using Test

using QuantumOptics

@testset "pauli" begin

@test_throws MethodError PauliBasis(1.4)

# Test conversion of unitary matrices to superoperators.
q2 = PauliBasis(2)
q3 = PauliBasis(3)
CZ = DenseOperator(q2, q2, diagm(0 => [1,1,1,-1]))
CZ_sop = SuperOperator(CZ)

# Test conversion of unitary matrices to superoperators.
@test diag(CZ_sop.data) ==  ComplexF64[1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1]
@test CZ_sop.basis_l == CZ_sop.basis_r == (q2, q2)

# Test conversion of superoperator to Pauli transfer matrix.
CZ_ptm = PauliTransferMatrix(CZ_sop)

# Test DensePauliTransferMatrix constructor.
@test_throws DimensionMismatch DensePauliTransferMatrix((q2, q2), (q3, q3), CZ_ptm.data)
@test DensePauliTransferMatrix((q2, q2), (q2, q2), CZ_ptm.data) == CZ_ptm

@test all(isapprox.(CZ_ptm.data[[1,30,47,52,72,91,117,140,166,185,205,210,227,256]], 1))
@test all(isapprox.(CZ_ptm.data[[106,151]], -1))

@test CZ_ptm == PauliTransferMatrix(ChiMatrix(CZ))

# Test construction of non-symmetric unitary.
CNOT = DenseOperator(q2, q2, diagm(0 => [1,1,0,0], 1 => [0,0,1], -1 => [0,0,1]))
CNOT_sop = SuperOperator(CNOT)
CNOT_chi = ChiMatrix(CNOT)
CNOT_ptm = PauliTransferMatrix(CNOT)

@test CNOT_sop.basis_l == CNOT_sop.basis_r == (q2, q2)
@test CNOT_chi.basis_l == CNOT_chi.basis_r == (q2, q2)
@test CNOT_ptm.basis_l == CNOT_ptm.basis_r == (q2, q2)

@test all(isapprox.(imag.(CNOT_sop.data), 0))
@test all(isapprox.(imag.(CNOT_chi.data), 0))
@test all(isapprox.(imag.(CNOT_ptm.data), 0))

@test all(isapprox.(CNOT_sop.data[[1,18,36,51,69,86,104,119,141,158,176,191,201,218,236,251]], 1))
@test all(isapprox.(CNOT_chi.data[[1,2,13,17,18,29,193,194,205,222]], 1))
@test all(isapprox.(CNOT_chi.data[[14,30,206,209,210,221]], -1))
@test all(isapprox.(CNOT_ptm.data[[1,18,47,64,70,85,108,138,153,183,205,222,227,244,]], 1))
@test all(isapprox.(CNOT_ptm.data[[123,168]], -1))

# Test DenseChiMatrix constructor.
@test_throws DimensionMismatch DenseChiMatrix((q2, q2), (q3, q3), CNOT_chi.data)
@test DenseChiMatrix((q2, q2), (q2, q2), CNOT_chi.data) == CNOT_chi

# Test equality and conversion among all three bases.
ident = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

IDENT = DenseOperator(q2, ident)

IDENT_sop = SuperOperator(IDENT)
IDENT_chi = ChiMatrix(IDENT)
IDENT_ptm = PauliTransferMatrix(IDENT)

@test ChiMatrix(IDENT_sop) == IDENT_chi
@test ChiMatrix(IDENT_ptm) == IDENT_chi
@test SuperOperator(IDENT_chi) == IDENT_sop
@test SuperOperator(IDENT_ptm) == IDENT_sop
@test PauliTransferMatrix(IDENT_sop) == IDENT_ptm
@test PauliTransferMatrix(IDENT_chi) == IDENT_ptm

# Test approximate equality and conversion among all three bases.
cphase = Complex{Float64}[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(1im*.6)]

CPHASE = DenseOperator(q2, cphase)

CPHASE_sop = SuperOperator(CPHASE)
CPHASE_chi = ChiMatrix(CPHASE)
CPHASE_ptm = PauliTransferMatrix(CPHASE)

@test isapprox(ChiMatrix(CPHASE_sop), CPHASE_chi)
@test isapprox(ChiMatrix(CPHASE_ptm), CPHASE_chi)
@test isapprox(SuperOperator(CPHASE_chi), CPHASE_sop)
@test isapprox(SuperOperator(CPHASE_ptm), CPHASE_sop)
@test isapprox(PauliTransferMatrix(CPHASE_sop), CPHASE_ptm)
@test isapprox(PauliTransferMatrix(CPHASE_chi), CPHASE_ptm)

# Test composition.
@test isapprox(ChiMatrix(CPHASE) * ChiMatrix(CNOT), ChiMatrix(CPHASE * CNOT))
@test isapprox(PauliTransferMatrix(CPHASE) * PauliTransferMatrix(CNOT), PauliTransferMatrix(CPHASE * CNOT))

end # testset
