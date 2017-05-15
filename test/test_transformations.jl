using Base.Test
using QuantumOptics

@testset "fock" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))
D(x::Ket, y::Ket) = norm(x-y)

# Test transformations
b_fock = FockBasis(20)
b_position = PositionBasis(-10, 10, 200)
Txn = transform(b_position, b_fock)
Tnx = transform(b_fock, b_position)
@test 1e-10 > D(dagger(Txn), Tnx)
@test 1e-10 > D(one(b_fock), Tnx*Txn)

x0 = 0.1
p0 = 0.3
α0 = (x0 + 1im*p0)/sqrt(2)
psi_n = coherentstate(b_fock, α0)
psi_x = gaussianstate(b_position, x0, p0, 1)
@test 1e-10 > D(psi_x, Txn*psi_n)

# Test different characteristic length
x0 = 0.0
p0 = 0.2
α0 = (x0 + 1im*p0)/sqrt(2)
σ0 = 0.7
Txn = transform(b_position, b_fock; x0=σ0)
Tnx = transform(b_fock, b_position; x0=σ0)
@test 1e-10 > D(dagger(Txn), Tnx)
@test 1e-10 > D(one(b_fock), Tnx*Txn)

psi_n = coherentstate(b_fock, α0)
psi_x = gaussianstate(b_position, x0/σ0, p0/σ0, σ0)
@test 1e-10 > D(psi_x, Txn*psi_n)



end # testset