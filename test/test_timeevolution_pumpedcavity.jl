using Base.Test
using QuantumOptics

@testset "timeevolution_pumpedcavity" begin

# System parameters
ω = 1.89 # Frequency of driving laser
ωc = 2.13 # Cavity frequency
η = 0.76 # Pump strength
κ = 0.34 # Decay rate

δc = ωc - ω # Detuning

# System basis and operators
b = FockBasis(50)

a = destroy(b)
at = create(b)
n = number(b)

Hint = δc*n + η*(a + at)

# Initial state
α0 = 0.3 - 0.5im
psi0 = coherentstate(b, α0)

# Analytic solution
function α(t, α0, δc, κ)
  Δ = δc - 1im*κ/2
  (α0 + η/Δ)*exp(-1im*Δ*t) - η/Δ
end

D(psi1::Ket, psi2::Ket) = abs(1 - abs2(dagger(psi1)*psi2))
D(rho1::DenseOperator, rho2::DenseOperator) = tracedistance(rho1, rho2)

T = [0:1.:3;]

# No decay
f_test(t, psi::Ket) = @test 1e-5 > D(psi, coherentstate(b, α(t, α0, δc, 0)))
f_test(t, rho::DenseOperator) = @test 1e-5 > tracedistance(rho, dm(coherentstate(b, α(t, α0, δc, 0))))

timeevolution.schroedinger(T, psi0, Hint; fout=f_test)
timeevolution.mcwf(T, psi0, Hint, []; fout=f_test)
timeevolution.master(T, psi0, Hint, []; fout=f_test)
timeevolution.master_h(T, psi0, Hint, []; fout=f_test)
timeevolution.master_nh(T, psi0, Hint, []; fout=f_test)

# No decay, rotating
f_test_td(t, psi::Ket) = @test 1e-5 > D(psi, coherentstate(b, α(t, α0, δc, 0)*exp(-1im*ω*t)))
f_test_td(t, rho::DenseOperator) = @test 1e-5 > D(rho, dm(coherentstate(b, α(t, α0, δc, 0)*exp(-1im*ω*t))))

f_H(t, psi) = ωc*n + η*(a*exp(1im*ω*t) + at*exp(-1im*ω*t))
f_HJ(t, rho) = (f_H(t, psi0), [], [])

timeevolution.schroedinger_dynamic(T, psi0, f_H; fout=f_test_td)
timeevolution.master_dynamic(T, psi0, f_HJ; fout=f_test_td)

# Decay
Hint_nh = Hint - 0.5im*κ*n
Γ = Matrix{Float64}(1,1)
Γ[1,1] = κ
J = [a]
Jdagger = [at]

f_test_decay(t, rho::DenseOperator) = @test 1e-5 > tracedistance(rho, dm(coherentstate(b, α(t, α0, δc, κ))))

timeevolution.master(T, psi0, Hint, J; Gamma=Γ, fout=f_test_decay)
timeevolution.master_h(T, psi0, Hint, J; Gamma=Γ, fout=f_test_decay)
timeevolution.master_nh(T, psi0, Hint_nh, J; Gamma=Γ, fout=f_test_decay)

# Decay, rotating
f_test_decay_dynamic(t, rho::DenseOperator) = @test 1e-5 > tracedistance(rho, dm(coherentstate(b, α(t, α0, δc, κ)*exp(-1im*ω*t))))

f_HJ_dynamic(t, rho) = (f_H(t, psi0), J, Jdagger)
f_HJ_nh_dynamic(t, rho) = (Hnh=f_H(t, psi0) - 0.5im*κ*n; (Hnh, dagger(Hnh), J, Jdagger))
f_HJ_nh_dynamic2(t, rho) = (Hnh=f_H(t, psi0) - 0.5im*κ*n; (Hnh, dagger(Hnh), [sqrt(κ)*a], [sqrt(κ)*at]))

timeevolution.master_dynamic(T, psi0, f_HJ_dynamic; Gamma=Γ, fout=f_test_decay_dynamic)
timeevolution.master_nh_dynamic(T, psi0, f_HJ_nh_dynamic; Gamma=Γ, fout=f_test_decay_dynamic)
timeevolution.master_nh_dynamic(T, psi0, f_HJ_nh_dynamic2; fout=f_test_decay_dynamic)

end # testset
