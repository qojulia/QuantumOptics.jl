using Base.Test
using QuantumOptics

@testset "stochastic_semiclassical" begin

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

rates = [0.1]
J = [sm]
C = [sm]
Jdagger = dagger.(J)
C .*= rates
Cdagger = dagger.(C)

u0 = Complex128[0.1, 0.5]
ψ_sc = semiclassical.State(ψ0, u0)
ρ_sc = dm(ψ_sc)

rates_mat = [0.1 0.05; 0.05 0.1]

dt = 1/100.0
T = [0:0.1:10;]
T_short = [0:dt:dt;]

# Function definitions for schroedinger_semiclassical
function fquantum(t, psi, u)
    return H
end
function fclassical(t, psi, u, du)
    du[1] = u[2]
    du[2] = sin(u[1])*cos(u[1])
end
function fquantum_stoch(t, psi, u)
    Hs
end
function fclassical_stoch(t, psi, u, du)
    du[1] = 0.2*u[1]
    du[2] = 0.2*u[2]
end
function fclassical_stoch2(t, psi, u, du)
    du[1,1] = 0.2*u[1]
    du[2,2] = 0.2*u[2]
end
function fclassical_stoch_ndiag(t, psi, u, du)
    du[1,1] = 0.2*u[1]
    du[1,2] = 0.1*u[1]
    du[2,3] = -0.1u[2]
end

# Function definitions for master_semiclassical
function fquantum_master(t, rho, u)
    H, J, Jdagger
end
function fstoch_q_master(t, rho, u)
    10 .*C, 10 .*Cdagger
end

# Test semiclassical schroedinger
tout, ψt_sc = stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical;
            fstoch_quantum=fquantum_stoch, dt=dt, normalize_state=true)
for ψ=ψt_sc
    @test norm(ψ.quantum) ≈ 1.0
end
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_classical=fclassical_stoch, dt=dt)
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_quantum=fquantum_stoch, fstoch_classical=fclassical_stoch2,
            noise_processes=1,
            noise_prototype_classical=zeros(Complex128, 2,2), dt=dt)
tout, ψt_sc = stochastic.schroedinger_semiclassical(T_short, ψ_sc, fquantum, fclassical;
            fstoch_classical=fclassical_stoch_ndiag,
            noise_prototype_classical=zeros(Complex128, 2, 3), dt=dt)

# Semiclassical master
tout, ρt = stochastic.master_semiclassical(T, ρ_sc, fquantum_master, fclassical;
            fstoch_quantum=fstoch_q_master, dt=dt)
tout, ρt = stochastic.master_semiclassical(T, ρ_sc, fquantum_master, fclassical;
            fstoch_classical=fclassical_stoch, dt=dt)
tout, ρt = stochastic.master_semiclassical(T, ψ_sc, fquantum_master, fclassical;
            fstoch_quantum=fstoch_q_master, fstoch_classical=fclassical_stoch2,
            noise_prototype_classical=zeros(Complex128, 2, 2), dt=dt)

# Test error messages
@test_throws ArgumentError stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical)
@test_throws ArgumentError stochastic.schroedinger_semiclassical(T, ψ_sc, fquantum, fclassical;
        fstoch_quantum=fquantum_stoch, fstoch_classical=fclassical_stoch)
@test_throws ArgumentError stochastic.master_semiclassical(T, ρ_sc, fquantum_master, fclassical)

end # testset
