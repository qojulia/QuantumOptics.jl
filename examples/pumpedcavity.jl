using QuantumOptics

# System parameters for a pumped decaying cavity
η = 0.9
κ = 1.

# System operators
basis = FockBasis(20)
a = destroy(basis)
aᵀ = create(basis)
n = number(basis)
H = η*(a+aᵀ)
J = [sqrt(κ)*a]

# Initial state
Ψ₀ = fockstate(basis, 10)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

# Integration options
Ntrajectories = 50
T = [0:0.1:10;]


# Master time evolution
tout, ρt_master = timeevolution.master(T, ρ₀, H, J)

# Averaged MCWF time evolution
ρt_average = Operator[ρ₀*0. for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J)
    for j=1:length(T)
        ρt_average[j] += (Ψt[j] ⊗ dagger(Ψt[j]))/Ntrajectories
    end
end

# Example trajectories
tout_example1, Ψt_example1 = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1),
                                        display_beforeevent=true,
                                        display_afterevent=true)

tout_example2, Ψt_example2 = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(2),
                                        display_beforeevent=true,
                                        display_afterevent=true)

tout_example3, Ψt_example3 = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(3),
                                        display_beforeevent=true,
                                        display_afterevent=true)


using PyCall
@pyimport matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(T, real(expect(n, ρt_master)))
plt.plot(T, real(expect(n, ρt_average)))
plt.xlabel("\$\\mathrm\{Time\} \$")
plt.ylabel("\$\\mathrm{Photon number}\$")

plt.subplot(2, 1, 2)
plt.plot(tout_example1, real(expect(n, Ψt_example1)))
plt.plot(tout_example2, real(expect(n, Ψt_example2)))
plt.plot(tout_example3, real(expect(n, Ψt_example3)))
plt.xlabel("\$\\mathrm\{Time\} \$")
plt.ylabel("\$\\mathrm{Photon number}\$")

plt.show()
