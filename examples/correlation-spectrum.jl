
using QuantumOptics
correlation = timecorrelations.correlation
corr2spec = timecorrelations.correlation2spectrum
using PyPlot

Nc = 10
κ = 1.0
n = 4
Δ = 5.0κ;

basis = FockBasis(Nc)

a = destroy(basis)
H = -Δ*dagger(a)*a
J = [sqrt(2κ)*a];

ρ₀ = fockstate(basis, n) ⊗ dagger(fockstate(basis, n))

dτ = 0.05
τmax = 1000
τ = [0:dτ:τmax;]
corr = correlation(τ, ρ₀, H, J, dagger(a), a);

ω, spec = corr2spec(τ, corr; normalize=true);

corr_an = n.*exp(-1.0im*Δ.*τ).*exp(-κ.*τ)
spec_an = 2n*κ./((Δ + ω).^2 + κ^2)
spec_an ./= maximum(spec_an);

clf()
figure(figsize=(9, 3))
subplot(121)
plot(ω, spec, label="numerical")
plot(ω, spec_an, ls="dashed", label="analytical")
xlabel("freq.")
ylabel("spectrum")
subplot(122)
plot(τ[1:200], real(corr[1:200]))
plot(τ[1:200], real(corr_an[1:200]), ls="dashed")
xlabel("time")
ylabel("correlation")
show()


