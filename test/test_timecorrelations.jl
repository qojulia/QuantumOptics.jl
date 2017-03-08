using Base.Test
using QuantumOptics

@testset "correlations" begin

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

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

Ja = embed(basis, 1, sqrt(γ)*sm)
Ja2 = embed(basis, 1, sqrt(0.5*γ)*sp)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Ja2, Jc]

Ψ₀ = basis_ket(spinbasis, 2) ⊗ fockstate(fockbasis, 5)
ρ₀ = Ψ₀⊗dagger(Ψ₀)

tspan = [0.:10.:100.;]

op = embed(basis, 1, sqrt(γ)*sz)
exp_values = correlations.correlation(tspan, ρ₀, H, J, dagger(op), op)

ρ₀ = Ψ₀⊗dagger(Ψ₀)

tout, exp_values2 = correlations.correlation(ρ₀, H, J, dagger(op), op; eps=1e-5)

@test length(exp_values) == length(tspan)
@test length(exp_values2) == length(tout)
@test norm(exp_values[1]-exp_values2[1]) < 1e-15
@test norm(exp_values[end]-exp_values2[end]) < 1e-4

op = embed(basis, 1, sqrt(γ)*sm)
omega, S = correlations.correlationspectrum(H, J, op)

end # testset
