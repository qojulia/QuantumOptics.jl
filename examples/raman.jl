
using QuantumOptics
using PyPlot

# Parameters
γ₃ = 1.
Ω = .5γ₃
Δ₂ = 5γ₃
Δ₃ = 0.0
tmax = 800/γ₃
dt = 0.1
tlist = [0:dt:tmax;];

# Basis and operators
b = NLevelBasis(3)
σ₁ = transition(b, 1, 2)
σ₃ = transition(b, 3, 2)
proj₂ = transition(b, 2, 2)
proj₃ = σ₃*dagger(σ₃);

# Hamiltonian and jump operators
H = Δ₂*proj₂ + Δ₃*proj₃ + Ω*(σ₁ + dagger(σ₁))
J = [sqrt(γ₃)*σ₃];

# Initial state
ψ₀ = nlevelstate(b, 1);

# Expectation values
tout = Float64[]
p1 = Float64[]
p2 = Float64[]
p3 = Float64[]
function calc_pops(t, ρ)
    push!(tout, t)
    push!(p1, real(expect(σ₁*dagger(σ₁), ρ)))
    push!(p2, real(expect(proj₂, ρ)))
    push!(p3, real(expect(proj₃, ρ)))
end;

# Time evolution
timeevolution.master(tlist, ψ₀, H, J; fout=calc_pops)

# Plots
figure(figsize=(6, 3))
plot(tout, p1, label="Initial ground state")
plot(tout, p2, "--", label="Excited state")
plot(tout, p3, label="Other ground state")
axis([0, tmax, 0, 1])
legend()
show()


