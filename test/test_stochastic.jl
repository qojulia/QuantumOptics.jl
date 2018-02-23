using Base.Test
using QuantumOptics

@testset "stochastic" begin

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
Js = [sm]
Jdagger = dagger.(J)
Js .*= rates
Jsdagger = dagger.(Js)

u0 = Complex128[0.1, 0.5]
ψ_sc = semiclassical.State(ψ0, u0)
ρ_sc = dm(ψ_sc)

rates_mat = [0.1 0.05; 0.05 0.1]

dt = 1/30.0
T = [0:0.1:1;]
T_short = [0:dt:dt;]

# Test equivalence of stochastic schroedinger phase noise and master dephasing
Ntraj = 100
ρ_avg = [0*ρ0 for i=1:length(T)]
for i=1:Ntraj
    t, ψt = stochastic.schroedinger(T, ψ0, H, Hs; dt=1e-3)
    ρ_avg += dm.(ψt)./Ntraj
end
tout, ρt = timeevolution.master(T, ρ0, H, [sz]; rates=[0.25γ^2])

for i=1:length(tout)
    @test tracedistance(ρ_avg[i], ρt[i]) < dt
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
function fstoch_3(t, psi)
    noise_op, noise_op
end

# Function definitions for master_dynamic
function fdeterm_master(t, rho)
    H, J, Jdagger
end
function fstoch1_master(t, rho)
    [zero_op], [zero_op], rates
end
function fstoch2_master(t, rho)
    Js, Jsdagger
end
function fstoch3_master(t, rho)
    Hs
end
function fstoch4_master(t, rho)
    J, Jdagger, rates
end

# Function definitions for schroedinger_semiclassical
function fquantum(t, psi, u)
    return H
end
function fclassical(t, psi, u, du)
    du[1] = 3*u[2]
    du[2] = 5*sin(u[1])*cos(u[1])
end
function fquantum_stoch(t, psi, u)
    10 .* Hs
end
function fclassical_stoch(t, psi, u, du)
    du[2] = 2*u[2]
end

# Function definitions for master_semiclassical
function fquantum_master(t, rho, u)
    H, J, Jdagger
end
function fstoch_q_master(t, rho, u)
    Js, Jsdagger
end
function fstoch_q_master2(t, rho, u)
    Js, Jsdagger, ones(length(Js))
end
function fstoch_J(t, rho, u)
    J, Jdagger
end

# Non-dynamic Schrödinger
tout, ψt4 = stochastic.schroedinger(T, ψ0, H, [zero_op, zero_op]; dt=dt)
tout, ψt3 = stochastic.schroedinger(T, ψ0, H, zero_op; dt=dt)
# Dynamic Schrödinger
tout, ψt1 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_1; dt=dt)
tout, ψt2 = stochastic.schroedinger_dynamic(T, ψ0, fdeterm, fstoch_2; dt=dt, noise_processes=3)

# Test equivalence to Schrödinger equation with zero noise
# Test sharp equality for same algorithms
@test ψt1 == ψt3
@test ψt2 == ψt4

tout, ψt_determ = timeevolution.schroedinger_dynamic(T, ψ0, fdeterm)
# Test approximate equality for different algorithms
for i=1:length(tout)
    @test norm(ψt1[i] - ψt2[i]) < dt
    @test norm(ψt1[i] - ψt_determ[i]) < dt
end

# Test master
tout, ρt_det = timeevolution.master(T, ψ0, H, J; rates=rates)
tout, ρt1 = stochastic.master(T, ψ0, H, J, 0 .*J; rates=rates, dt=dt)
tout, ρt2 = stochastic.master(T, ρ0, LazyProduct(H, one(H)), sqrt.(rates).*J, 0 .* J; Hs=0 .* Hs, dt=dt)
tout, ρt3 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch1_master; rates=rates, dt=dt)
for i=1:length(tout)
    @test tracedistance(ρt1[i], ρt_det[i]) < dt
    @test tracedistance(ρt2[i], ρt_det[i]) < dt
    @test tracedistance(ρt3[i], ρt_det[i]) < dt
end

# Test remaining function calls for short times to test whether they work in principle
# Schroedinger
tout, ψt = stochastic.schroedinger(T_short, ψ0, H, noise_op; dt=dt)
tout, ψt = stochastic.schroedinger_dynamic(T_short, ψ0, fdeterm, fstoch_3; dt=dt)

# Master
tout, ρt = stochastic.master(T_short, ρ0, H, J, 0.*J; dt=dt)
tout, ρt = stochastic.master(T_short, ψ0, H, [sm, sm], [sm, sm]; rates=rates_mat, dt=dt)
tout, ρt = stochastic.master(T_short, ρ0, H, J, J; rates_s=[0.1], Hs=Hs, dt=dt)

# Test master dynamic
tout, ρt = stochastic.master_dynamic(T_short, ψ0, fdeterm_master, fstoch1_master; dt=dt, noise_processes=1)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_H=fstoch3_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_J=fstoch4_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_J=fstoch2_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch4_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch2_master, dt=dt)

# Test semiclassical
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_quantum=fquantum_stoch, dt=dt)
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_classical=fclassical_stoch, noise_processes=1, dt=dt)
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_quantum=fquantum_stoch, fstoch_classical=fclassical_stoch, dt=dt)

# Semiclassical master
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical; fstoch_quantum=fstoch_q_master, dt=dt, noise_processes=1)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical; fstoch_classical=fclassical_stoch, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ψ_sc, fquantum_master, fclassical;
            fstoch_quantum=fstoch_q_master2, fstoch_classical=fclassical_stoch, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ψ_sc, fquantum_master, fclassical;
            fstoch_quantum=fstoch_q_master, fstoch_classical=fclassical_stoch, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical;
            fstoch_classical=fclassical_stoch,
            fstoch_H=fquantum_stoch, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical;
            fstoch_classical=fclassical_stoch,
            fstoch_J=fstoch_J, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical;
            fstoch_classical=fclassical_stoch,
            fstoch_J=fstoch_q_master2, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical;
            fstoch_quantum=fstoch_q_master2, dt=dt)
tout, ρt = stochastic.master_semiclassical(T_short, ρ_sc, fquantum_master, fclassical;
            fstoch_H=fquantum_stoch, fstoch_J=fstoch_J, dt=dt)

# Test error messages
@test_throws ArgumentError stochastic.master(T, ψ0, H, [sm, sm], [sm, sm]; rates_s=rates_mat, dt=dt)
@test_throws ArgumentError stochastic.master_dynamic(T, ψ0, fdeterm_master, fstoch1_master; rates_s=rates_mat, dt=dt)
@test_throws ArgumentError stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical)
@test_throws ArgumentError stochastic.master_semiclassical(T, ρ_sc, fquantum_master, fclassical)
@test_throws ArgumentError stochastic.master_semiclassical(T, ρ_sc, fquantum_master, fclassical;
            fstoch_classical=fclassical_stoch, rates_s=rates_mat)

end # testset
