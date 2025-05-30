@testitem "timeevolution_twolevel" begin
using QuantumOptics
using LinearAlgebra

basis = SpinBasis(1//2)

# Random 2 level Hamiltonian
a1 = 0.5
a2 = 1.9
c = 1.3
d = -4.7

data = [a1 c-1im*d; c+1im*d a2]
H = DenseOperator(basis, data)

a = (a1 + a2)/2
b = (a1 - a2)/2
r = [c d b]

sigma_r = c*sigmax(basis) + d*sigmay(basis) + b*sigmaz(basis)

U(t) = exp(-1im*a*t)*(cos(norm(r)*t)*one(basis) - 1im*sin(norm(r)*t)*sigma_r/norm(r))

# Random initial state
psi0 = randstate(basis)
T = [0:0.5:1;]

f(t, psi::Ket) = @test 1e-5 > norm(psi - U(t)*psi0)
timeevolution.schroedinger(T, psi0, H; fout=f)
timeevolution.mcwf(T, psi0, H, []; fout=f)

f(t, rho::DenseOpType) = @test 1e-5 > tracedistance(rho, dm(U(t)*psi0))
timeevolution.master(T, psi0, H, []; fout=f)

end # testset
