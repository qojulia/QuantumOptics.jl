using Base.Test
using QuantumOptics
try
    using CollectiveSpins
catch e
    println("CollectiveSpins.jl not found - skipping test.")
    quit()
end


@testset "correlationexpansion_mpc" begin

cs = CollectiveSpins
ce = correlationexpansion

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_general(full(op1), full(op2)))


N = 4
a = 0.54
γ = 1.
e_dipole = [0, 0, 1]
T = [0:0.1:0.5;]
phi = 2*pi*rand(N)
theta = pi*rand(N)

system = SpinCollection(cs.geometry.chain(a, N), e_dipole; gamma=γ)
spins = system.spins
b = cs.quantum.basis(system)

S2 = ce.masks(N, 2)
# S3 = ce.masks(N, 3)
# S4 = ce.masks(N, 4)

# Build Hamiltonian and Jump operators
spinbasis = SpinBasis(1//2)
sigmap = spin.sigmap(spinbasis)
sigmam = spin.sigmam(spinbasis)
I_spin = identityoperator(spinbasis)

H = LazyTensor[]
for i=1:N, j=1:N
    if i==j
        continue
    end
    Ωij = cs.interaction.Omega(spins[i].position, spins[j].position, system.polarization, system.gamma)
    push!(H, Ωij*LazyTensor(b, [i, j], [sigmap, sigmam]))
end
H = LazySum(H...)

J = LazyTensor[LazyTensor(b, i, sigmam) for i=1:N]
Γ = cs.interaction.GammaMatrix(system)

H_ = cs.quantum.Hamiltonian(system)
Γ_, J_ = cs.quantum.JumpOperators(system)
@test 1e-15 > D(H_, H)
for i=1:N
    @test 1e-15 > D(J_[i], J[i])
end

# Master
psi0 = cs.quantum.blochstate(phi, theta)
rho0 = psi0⊗dagger(psi0)
# tout, rho_t = cs.quantum.timeevolution(T, system, rho0)
tout, rho_t = timeevolution.master_h(T, rho0, full(H), [full(j) for j in J]; Gamma=Γ)

ce0 = ce.approximate(rho0, ce.masks(N))
tout, ce_t = ce.master(T, ce0, H, J; Gamma=Γ)

for (rho, rho_ce) in zip(rho_t, ce_t)
    @test 1e-5 > D(rho, rho_ce)
end


# Meanfield
state0 = cs.meanfield.blochstate(phi, theta)
tout, state_mf_t = cs.meanfield.timeevolution(T, system, state0)

ce0 = ce.approximate(rho0)
tout, ce_t = ce.master(T, ce0, H, J; Gamma=Γ)

for (state_mf, rho_ce, rho) in zip(state_mf_t, ce_t, rho_t)
    rho_mf = cs.meanfield.densityoperator(state_mf)
    d1 = abs(tracedistance_general(full(rho), full(rho_mf)))
    d2 = abs(tracedistance_general(full(rho), full(rho_ce)))
    @test d2 < 2*d1
end


# Meanfield + Correlations
state0 = cs.mpc.blochstate(phi, theta)
tout, state_mpc_t = cs.mpc.timeevolution(T, system, state0)

ce0 = ce.approximate(rho0, S2)
tout, ce_t = ce.master(T, ce0, H, J; Gamma=Γ)

for (state_mpc, rho_ce, rho) in zip(state_mf_t, ce_t, rho_t)
    rho_mpc = cs.mpc.densityoperator(state_mpc)
    d1 = abs(tracedistance_general(full(rho), full(rho_mpc)))
    d2 = abs(tracedistance_general(full(rho), full(rho_ce)))
    @test d2 < 2*d1
end

end # testset