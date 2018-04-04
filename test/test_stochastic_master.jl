using Base.Test
using QuantumOptics

@testset "stochastic_master" begin

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

rates_mat = [0.1 0.05; 0.05 0.1]

dt = 1/30.0
T = [0:0.1:1;]
T_short = [0:dt:dt;]

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
J2 = [J[1], J[1]]
J2dagger = dagger.(J2)
rates2 = [rates[1], rates[1]]
function fstoch5_master(t, rho)
    J2, J2dagger, rates2
end

# Test master
tout, ρt_det = timeevolution.master(T, ψ0, H, J; rates=rates)
tout, ρt1 = stochastic.master(T, ψ0, H, J, 0 .*J; rates=rates, dt=dt)
tout, ρt2 = stochastic.master(T, ρ0, LazyProduct(H, one(H)), sqrt.(rates).*J, 0 .* J; Hs=0 .* Hs, dt=dt)
tout, ρt3 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch1_master; rates=rates, dt=dt)
tout, ρt4 = stochastic.master(T, ψ0, H, J, 0 .*J; rates=rates, dt=dt, nonlinear=false)
tout, ρt5 = stochastic.master(T, ρ0, LazyProduct(H, one(H)), sqrt.(rates).*J, 0 .* J; Hs=0 .* Hs, dt=dt, nonlinear=false)
tout, ρt6 = stochastic.master_dynamic(T, ρ0, fdeterm_master, fstoch1_master; rates=rates, dt=dt, nonlinear=false)
for i=1:length(tout)
    @test tracedistance(ρt1[i], ρt_det[i]) < dt
    @test tracedistance(ρt2[i], ρt_det[i]) < dt
    @test tracedistance(ρt3[i], ρt_det[i]) < dt
    @test tracedistance(ρt4[i], ρt_det[i]) < dt
    @test tracedistance(ρt5[i], ρt_det[i]) < dt
    @test tracedistance(ρt6[i], ρt_det[i]) < dt
end

# Test remaining function calls for short times to test whether they work in principle
tout, ρt = stochastic.master(T_short, ρ0, H, J, 0.*J; dt=dt)
tout, ρt = stochastic.master(T_short, ψ0, H, [sm, sm], [sm, sm]; rates=rates_mat, dt=dt)
tout, ρt = stochastic.master(T_short, ρ0, H, J, J; rates_s=[0.1], Hs=Hs, dt=dt)

# Linear version
tout, ρt = stochastic.master(T_short, ρ0, H, J, 0.*J; dt=dt, nonlinear=false)
tout, ρt = stochastic.master(T_short, ψ0, H, [sm, sm], [sm, sm]; rates=rates_mat, dt=dt, nonlinear=false)
tout, ρt = stochastic.master(T_short, ρ0, H, J, J; rates_s=[0.1], Hs=Hs, dt=dt, nonlinear=false)

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
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; rates_s=[1.0], dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch2_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_H=fstoch3_master, dt=dt)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_J=fstoch5_master, dt=dt)

# Linear version
tout, ρt = stochastic.master_dynamic(T_short, ψ0, fdeterm_master, fstoch1_master; dt=dt, noise_processes=1, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_H=fstoch3_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_J=fstoch4_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master; fstoch_J=fstoch2_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch4_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch2_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch2_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_H=fstoch3_master, fstoch_J=fstoch4_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_J=fstoch4_master, dt=dt, nonlinear=false)
tout, ρt = stochastic.master_dynamic(T_short, ρ0, fdeterm_master, fstoch4_master;
            fstoch_H=fstoch3_master, dt=dt, nonlinear=false)


# Test error messages
@test_throws ArgumentError stochastic.master(T, ψ0, H, [sm, sm], [sm, sm]; rates_s=rates_mat, dt=dt)
@test_throws ArgumentError stochastic.master_dynamic(T, ψ0, fdeterm_master, fstoch1_master; rates_s=rates_mat, dt=dt)

end # testset
