@testitem "test_timecorrelations" begin
using Test
using QuantumOptics
using LinearAlgebra

@testset "timecorrelations" begin

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

# time-dependent Hamiltonian in rotating frame of cavity mode,
# which should not change the time correlation for spin operators.
f_HJ(t,rho) = [ Ha + exp(im*ωc*t) * sm ⊗ create(fockbasis) + exp(-im*ωc*t) * sp ⊗ destroy(fockbasis), J, dagger.(J) ]

Ψ₀ = basisstate(spinbasis, 2) ⊗ fockstate(fockbasis, 5)
ρ₀ = dm(Ψ₀)

tspan = [0.:10:100.;]

op = embed(basis, 1, sqrt(γ)*sz)
exp_values = timecorrelations.correlation(tspan, ρ₀, H, J, dagger(op), op)

ρ₀ = dm(Ψ₀)

tout, exp_values2 = timecorrelations.correlation(ρ₀, H, J, dagger(op), op; tol=1e-5)
  
exp_values3 = timecorrelations.correlation_dynamic(tspan, ρ₀, f_HJ, dagger(op), op)

@test length(exp_values) == length(tspan)
@test length(exp_values2) == length(tout)
@test length(exp_values3) == length(tspan)
@test norm(exp_values[1]-exp_values2[1]) < 1e-15
@test norm(exp_values[end]-exp_values2[end]) < 1e-4
@test all(norm.(exp_values .- exp_values3) .< 1e-4)
  
n = length(tspan)
omega_sample = mod(n, 2) == 0 ? [-n/2:n/2-1;] : [-(n-1)/2:(n-1)/2;]
omega_sample .*= 2pi/tspan[end]
omega, S = timecorrelations.spectrum(omega_sample, H, J, op; rho_ss=ρ₀)

omega2, S2 = timecorrelations.spectrum(H, J, op; tol=1e-3)
@test length(omega2) == length(S2)

omegaFFT, SFFT = timecorrelations.correlation2spectrum(tspan, exp_values)

@test omega_sample == omegaFFT && S == SFFT

@test_throws ArgumentError timecorrelations.correlation2spectrum(tout, exp_values)

tspan[5] = tspan[4]
@test_throws ArgumentError timecorrelations.correlation2spectrum(tspan, exp_values)

# tests kwarg rates
J5 = [embed(basis, 1, sm), embed(basis, 1, sp), embed(basis, 2, destroy(fockbasis))]
rates5 = [γ, 0.5γ, κ]
exp_values5 = timecorrelations.correlation(tspan, ρ₀, H, J5, dagger(op), op; rates=rates5)
@test abs(exp_values[end] - exp_values5[end] ) < 1e-8

omega5, S5 = timecorrelations.spectrum(omega_sample, H, J5, op;  rho_ss=ρ₀, rates=rates5)
@test abs(sum(S .- S5)) < 1e-8

omega5_2, S5_2 = timecorrelations.spectrum(H, J5, op; rates=rates5, tol=1e-3)
@test abs(sum(S2 .- S5_2)) < 1e-8

end # testset
end