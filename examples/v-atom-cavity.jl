using QuantumOptics
using PyPlot

"""
Example of a V-scheme three-level atom, where both transitions have the same
frequency and are resonantly coupled to a cavity mode. One transition is
resonantly driven from the side. The cavity is damped and the upper atomic
levels weakly decay with the same rate.
"""

# Parameters
Nc = 10
κ = 1.
Ω = κ/2.
g = κ
γ = κ/10.
dt = 0.1
tmax = 100/κ
T = [0:dt:tmax;]

# Bases and operators
bc = FockBasis(Nc)
ba = NLevelBasis(3) # Three-level atom: 1 is the ground state
idc = identityoperator(bc)
ida = identityoperator(ba)
a = destroy(bc) ⊗ ida
σ₂ = idc ⊗ transition(ba, 1, 2)
σ₃ = idc ⊗ transition(ba, 1, 3)

# Hamiltonian and jump operators
H_int = g*(a*dagger(σ₂ + σ₃) + dagger(a)*(σ₂ + σ₃))
H_pump = Ω*(σ₂ + dagger(σ₂))
H = H_int + H_pump

J = [sqrt(2κ)*a, sqrt(γ)*σ₂, sqrt(γ)*σ₃]

# Initial state
ψ₀ = fockstate(bc, 0) ⊗ nlevelstate(ba, 1)

# Time evolution
tout, ρt = timeevolution.master(T, ψ₀, H, J)

# Expectation values: photon number and level occupation
n = real(expect(dagger(a)*a, ρt))
p1 = real(expect(idc ⊗ transition(ba, 1, 1), ρt))
p2 = real(expect(dagger(σ₂)*σ₂, ρt))
p3 = real(expect(dagger(σ₃)*σ₃, ρt))

# Plots
clf()

subplot(211)
plot(tout, n, label="n")
legend()

subplot(212)
plot(tout, p1, "k--", label="Ground state")
plot(tout, p2, "b", label="1st excited state (driven)")
plot(tout, p3, "r", label="2nd excited state")
axis([0, tmax, 0, 1])
legend()

show()
