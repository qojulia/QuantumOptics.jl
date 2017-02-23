using Base.Test
using QuantumOptics

@testset "subspace" begin

b = FockBasis(3)

u = Ket[fockstate(b, 1), fockstate(b, 2)]
v = Ket[fockstate(b, 2), fockstate(b, 1)]

bu = SubspaceBasis(u)
bv = SubspaceBasis(v)

T1 = projector(bu, b)
T2 = projector(bv, b)
T12 = projector(bu, bv)

state = fockstate(b, 2)
state_u = Ket(bu, [0, 1])
state_v = Ket(bv, [1., 0])

@test T1*state == state_u
@test T2*state == state_v


state_v = Ket(bv, [1, -1])
state_u = Ket(bu, [-1, 1])

@test T12*state_v == state_u

u2 = Ket[1.5*fockstate(b, 1), fockstate(b, 1) + fockstate(b, 2)]
bu2_orth = subspace.orthonormalize(SubspaceBasis(u))
@test bu2_orth.basisstates == bu.basisstates

end # testset
