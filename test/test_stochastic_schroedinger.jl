using Test
using QuantumOptics
using LinearAlgebra
import StochasticDiffEq

@testset "stochastic_schroedinger" begin

b_spin = SpinBasis(1//2)
sz = sigmaz(b_spin)
sm = sigmam(b_spin)
sp = sigmap(b_spin)
zero_op = 0*sz
γ = 0.5
noise_op = 0.5γ*sz

H = γ*(sp + sm)
Hs = [noise_op]

ψ0 = spindown(b_spin)
ρ0 = dm(ψ0)

dt = 1/30.0
T = [0:1.0:100;]
T_short = [0:dt:dt;]

# Test equivalence of stochastic schroedinger phase noise and master dephasing
Ntraj = 100
ρ_avg = [0*ρ0 for i=1:length(T)]
for i=1:Ntraj
    t, ψt = stochastic.schroedinger(T, ψ0, H, Hs; dt=dt,
        alg=StochasticDiffEq.EulerHeun())
    ρ_avg += dm.(ψt)./Ntraj
end
tout, ρt = timeevolution.master(T, ρ0, H, [sz]; rates=[0.25γ^2])

for i=1:length(tout)
    @test tracedistance(ρ_avg[i], ρt[i]) < 5dt
end

# Function definitions for schroedinger_dynamic
function fdeterm(t, psi)
    H
end
function fstoch_1(t, psi)
    [zero_op]
end
function fstoch_2(t, psi)
    [zero_op, zero_op, zero_op]
end

# Non-dynamic Schrödinger
tout, ψt4 = stochastic.schroedinger(T, ψ0, H, [zero_op, zero_op]; dt=0.1dt)
# Dynamic Schrödinger
tout, ψt1 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_1; dt=0.1dt)
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_2; noise_processes=3, dt=0.1dt)

# Test equivalence to Schrödinger equation with zero noise
# Test sharp equality for same algorithms
@test ψt1 == ψt2 == ψt4

tout, ψt_determ = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm)
# Test approximate equality for different algorithms
for i=1:length(tout)
    @test norm(ψt1[i] - ψt2[i]) < 3dt
    @test norm(ψt1[i] - ψt_determ[i]) < 10dt
end

# Test remaining function calls for short times to test whether they work in principle
tout, ψt1 = stochastic.schroedinger(T, ψ0, H, noise_op; dt=dt,
        normalize_state=true,
        alg=StochasticDiffEq.EulerHeun())
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_1; dt=dt,
        normalize_state=true)

for i=1:length(T)
    @test norm(ψt1[i]) ≈ 1.0
    @test norm(ψt2[i]) ≈ 1.0
end

end # testset
