using Base.Test
using QuantumOptics

@testset "steadystate" begin

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1
η = 1.5

T = Float64[0.,1.]


fockbasis = FockBasis(10)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint
Hdense = full(H)

Ja = embed(basis, 1, sqrt(γ)*sm)
Ja2 = embed(basis, 1, sqrt(0.5*γ)*sp)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jdense = map(full, J)

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 2)
ρ₀ = Ψ₀⊗dagger(Ψ₀)
ψ0_p = fockstate(fockbasis, 0)
ρ0_p = ψ0_p ⊗ dagger(ψ0_p)

tout, ρt = timeevolution.master([0,100], ρ₀, Hdense, Jdense; reltol=1e-7)

ρss = steadystate.master(Hdense, Jdense; eps=1e-4)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.eigenvector(Hdense, Jdense)
@test tracedistance(ρss, ρt[end]) < 1e-6

ρss = steadystate.eigenvector(H, J)
@test tracedistance(ρss, ρt[end]) < 1e-3


# Compute steady-state photon number of a driven cavity (analytically: η^2/κ^2)
Hp = η*(destroy(fockbasis) + create(fockbasis))
Jp = [sqrt(2κ)*destroy(fockbasis)]
n_an = η^2/κ^2

ρss = steadystate.master(Hp, Jp; rho0=ρ0_p, eps=1e-4)
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3

ρss = steadystate.eigenvector(Hp, Jp)
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3

ρss = steadystate.eigenvector(full(Hp), map(full, Jp))
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3


# Test error messages
@test_throws ErrorException steadystate.eigenvector(sx, [sm])

end # testset
