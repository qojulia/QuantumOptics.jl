@testitem "stochastic_master" begin
using Test
using QuantumOptics
using LinearAlgebra

b_spin = SpinBasis(1//2)
sz = sigmaz(b_spin)
sm = sigmam(b_spin)
sp = sigmap(b_spin)
zero_op = 0*sz
γ = 0.1
noise_op = 0.5γ*sz

H = γ*(sp + sm)
Hs = [noise_op]

ψ0 = spindown(b_spin)

ρ0 = dm(ψ0)
rates = [0.1]
J = [sm]
C = [sm]
Jdagger = dagger.(J)
C .*= rates
Cdagger = dagger.(C)

dt = 1/30.0
T = [0:0.1:1;]
T_short = [0:dt:dt;]

# Function definitions for master_dynamic
function fdeterm_master(t, rho)
    H, J, Jdagger
end
function fstoch_master(t, rho)
    C, Cdagger
end

# Test master
tout, ρt_det = timeevolution.master(T, ψ0, H, J; rates=rates)
tout, ρt1 = stochastic.master(T, ψ0, H, J, 0 .*J; rates=rates, dt=dt)
tout, ρt2 = stochastic.master(T, ρ0, LazyProduct(H, one(H)), sqrt.(rates).*J, 0 .* J; dt=dt)
tout, ρt3 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch_master; rates=rates, dt=dt)
for i=1:length(tout)
    @test tracedistance(ρt1[i], ρt_det[i]) < dt
    @test tracedistance(ρt2[i], ρt_det[i]) < dt
    @test tracedistance(ρt3[i], ρt_det[i]) < dt
end

# Test remaining function calls for short times to test whether they work in principle
rates_mat = [0.1 0.05; 0.05 0.1]
tout, ρt = stochastic.master(T_short, ψ0, H, [sm, sm], [sm, sm]; rates=rates_mat, dt=dt)

# Test master dynamic
tout, ρt = stochastic.master_dynamic(T_short, ψ0, fdeterm_master, fstoch_master; noise_processes=1, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch_master, dt=dt)

end # testset
