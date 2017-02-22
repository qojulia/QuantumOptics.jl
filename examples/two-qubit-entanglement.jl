
using QuantumOptics
using PyPlot

# Parameters
Ω = 0.5
t = [0:0.1:10;];

# Hamiltonian
b = SpinBasis(1//2)
H = Ω*(sigmap(b) ⊗ sigmam(b) + sigmam(b) ⊗ sigmap(b));

ψ₀ = spindown(b) ⊗ spinup(b)
tout, ψₜ = timeevolution.schroedinger(t, ψ₀, H);

# Reduced density matrix
ρ_red = [ptrace(ψ ⊗ dagger(ψ), 1) for ψ=ψₜ]
S = [entropy_vn(ρ)/log(2) for ρ=ρ_red];

figure(figsize=(7, 3.5))
plot(tout, S)
show()


