using QuantumOptics
using LinearAlgebra
using Test


# Tests for the master_h_heisenberg function
# ==========================================
# We check that time evolution of <Ztot> is the same in the Schrödinger and Heisenberg
# pictures for a time reversal symmetry breaking Hamiltonian with non hermitian jump operators.


@testset "master_h_heisenberg" begin

N = 4
tmax = 10.0
dt = 0.1
b = SpinBasis(1//2)
basis = tensor([b for _=1:N]...)
sx(i) = embed(basis, i, sigmax(b))
sy(i) = embed(basis, i, sigmay(b))
sz(i) = embed(basis, i, sigmaz(b))


# time reversal symmetry breaking Hamiltonian
function ising(N)
    H = sum(sz(i) * sz(i+1) for i in 1:N-1)
    H += -sum(sx(i) for i in 1:N)
    H += -sum(sz(i) * sx(i+1)* sy(i+2) for i in 1:N-2)
    return H
end

Ztot(N) = sum(sz(i) for i in 1:N-1)

H = ising(N)

jump_ops = [sz(i)-im*sy(i) for i in 1:N]

ψ0 = tensor([spinup(b) for _=1:N]...)
ρ0 = ψ0 ⊗ dagger(ψ0)
times = 0:dt:tmax

A0 = Ztot(N)

fout_s(t, ρ) = real(expect(A0, ρ)) # shrödinger picture observable
fout_h(t, A) = real(expect(A, ρ0)) # heisenberg picture observable

n = length(jump_ops)

# test with no rates
tout_s, exp_val_s = timeevolution.master_h(times, ρ0, H, jump_ops; fout=fout_s)
tout_h, exp_val_h = timeevolution.master_h_heisenberg(times, A0, H, jump_ops; fout=fout_h)
@test norm(exp_val_s - exp_val_h) < 1e-5

# test with diagonal rates
rates = rand(n)*0.05
tout_s, exp_val_s = timeevolution.master_h(times, ρ0, H, jump_ops; fout=fout_s, rates=rates)
tout_h, exp_val_h = timeevolution.master_h_heisenberg(times, A0, H, jump_ops; fout=fout_h, rates=rates)
@test norm(exp_val_s - exp_val_h) < 1e-5

# test with rates matrix
rates = rand(n, n)*0.05
tout_s, exp_val_s = timeevolution.master_h(times, ρ0, H, jump_ops; fout=fout_s, rates=rates)
tout_h, exp_val_h = timeevolution.master_h_heisenberg(times, A0, H, jump_ops; fout=fout_h, rates=rates)
@test norm(exp_val_s - exp_val_h) < 1e-5

# uncomment to plot the results
# using Plots
# p = plot(tout_s,exp_val_s)
# plot!(p, tout_h, exp_val_h)
# display(p)
# sleep(100)
end
