using QuantumOptics
using PyPlot

"""
Example that illustrates the concept of a Raman transition. A Λ-scheme
three-level atom is prepared in one of two ground states (state 1 in the
following code) and is driven by a laser that is far-detuned from the
transition to the excited state. It spontaneously decays into the other ground
state.
"""

# Parameters
γ₃ = 1.
Ω = .5γ₃
Δ₂ = 5γ₃
Δ₃ = 0.0
tmax = 800/γ₃
dt = 0.1
tlist = [0:dt:tmax;]

# Basis and operators
b = NLevelBasis(3)
σ₁ = transition(b, 1, 2)
σ₃ = transition(b, 3, 2)
proj₂ = transition(b, 2, 2)
proj₃ = σ₃*dagger(σ₃)

# Hamiltonian and jump operators
H = Δ₂*proj₂ + Δ₃*proj₃ + Ω*(σ₁ + dagger(σ₁))
J = [sqrt(γ₃)*σ₃]

# Initial state
ψ₀ = nlevelstate(b, 1)

# Time evolution
tout, ρₜ = timeevolution.master(tlist, ψ₀, H, J)

# Expectation values
p1 = real(expect(σ₁*dagger(σ₁), ρₜ))
p2 = real(expect(proj₂, ρₜ))
p3 = real(expect(proj₃, ρₜ))

# Plots
clf()
plot(tout, p1, "b", label="Initial ground state")
plot(tout, p2, "k--", label="Excited state")
plot(tout, p3, "r", label="Other ground state")
axis([0, tmax, 0, 1])
legend()
show()
