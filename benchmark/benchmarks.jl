using BenchmarkTools
using QuantumOptics
using OrdinaryDiffEq
using StochasticDiffEq
using LinearAlgebra
using PkgBenchmark

const SUITE = BenchmarkGroup()

# =============================================================================
# Time evolution benchmarks (ODE/SDE solvers on QO types vs base array types)
# =============================================================================

prob_list = ("schroedinger", "master", "stochastic_schroedinger", "stochastic_master")
for prob in prob_list
    SUITE[prob] = BenchmarkGroup([prob])
    for type in ("qo types", "base array types")
        SUITE[prob][type] = BenchmarkGroup()
    end
end

function bench_schroedinger(dim; pure=true)
    b = SpinBasis(dim)
    t₀, t₁ = (0.0, pi)
    H = sigmax(b)
    psi0 = spindown(b)
    if pure
        obj = psi0.data
        Hobj = H.data
    else
        obj = psi0
        Hobj = H
    end
    schroed!(dpsi, psi, p, t) = timeevolution.dschroedinger!(dpsi, Hobj, psi)
    prob = ODEProblem(schroed!, obj, (t₀, t₁))
end

function bench_master(dim; pure=true)
    b = SpinBasis(dim)
    t₀, t₁ = (0.0, pi)
    H = sigmax(b)
    psi0 = spindown(b)
    J = sigmam(b)
    rho0 = dm(psi0)
    rates = [0.3]
    if pure
        obj = rho0.data
        Jobj, Jdag = (J.data, dagger(J).data)
        Hobj = H.data
    else
        obj = rho0
        Jobj, Jdag = (J, dagger(J))
        Hobj = H
    end
    master!(drho, rho, p, t) = timeevolution.dmaster_h!(drho, Hobj, [Jobj], [Jdag], rates, rho, copy(obj))
    prob = ODEProblem(master!, obj, (t₀, t₁))
end

function bench_stochastic_schroedinger(dim; pure=true)
    b = SpinBasis(dim)
    t₀, t₁ = (0.0, pi)
    H = sigmax(b)
    Hs = sigmay(b)
    psi0 = spindown(b)
    if pure
        obj = psi0.data
        Hobj = H.data
        Hsobj = Hs.data
    else
        obj = psi0
        Hobj = H
        Hsobj = Hs
    end
    schroed!(dpsi, psi, p, t) = timeevolution.dschroedinger!(dpsi, Hobj, psi)
    stoch_schroed!(dpsi, psi, p, t) = timeevolution.dschroedinger!(dpsi, Hsobj, psi)
    prob = SDEProblem(schroed!, stoch_schroed!, obj, (t₀, t₁))
end

function bench_stochastic_master(dim; pure=true)
    b = SpinBasis(dim)
    t₀, t₁ = (0.0, pi)
    H = sigmax(b)
    Hs = sigmay(b)
    psi0 = spindown(b)
    J = sigmam(b)
    rho0 = dm(psi0)
    rates = [0.3]
    if pure
        obj = rho0.data
        Jobj, Jdag = (J.data, dagger(J).data)
        Hobj = H.data
        Hsobj = Hs.data
    else
        obj = rho0
        Jobj, Jdag = (J, dagger(J))
        Hobj = H
        Hsobj = Hs
    end
    master!(drho, rho, p, t) = timeevolution.dmaster_h!(drho, Hobj, [Jobj], [Jdag], rates, rho, copy(obj))
    stoch_master!(drho, rho, p, t) = timeevolution.dmaster_h!(drho, Hsobj, [Jobj], [Jdag], rates, rho, copy(obj))
    prob = SDEProblem(master!, stoch_master!, obj, (t₀, t₁))
end

for dim in (1//2, 20//1, 50//1, 100//1)
    for solver in zip(("schroedinger", "master"), (:(bench_schroedinger), :(bench_master)))
        name, bench = (solver[1], solver[2])
        SUITE[name]["base array types"][string(dim)] = @benchmarkable solve(prob, DP5(); save_everystep=false) setup=(prob=eval($bench)($dim; pure=true))
        SUITE[name]["qo types"][string(dim)] = @benchmarkable solve(prob, DP5(); save_everystep=false) setup=(prob=eval($bench)($dim; pure=false))
    end
    for solver in zip(("stochastic_schroedinger", "stochastic_master"), (:(bench_stochastic_schroedinger), :(bench_stochastic_master)))
        name, bench = (solver[1], solver[2])
        SUITE[name]["base array types"][string(dim)] = @benchmarkable solve(prob, EM(), dt=1/100; save_everystep=false) setup=(prob=eval($bench)($dim; pure=true))
        SUITE[name]["qo types"][string(dim)] = @benchmarkable solve(prob, EM(), dt=1/100; save_everystep=false) setup=(prob=eval($bench)($dim; pure=false))
    end
end

# =============================================================================
# Operator algebra benchmarks
# =============================================================================

SUITE["operators"] = BenchmarkGroup(["operators"])

# Dense-dense addition
SUITE["operators"]["addition_dense_dense"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op1 = DenseOperator(b, rand(ComplexF64, N, N))
    op2 = DenseOperator(b, rand(ComplexF64, N, N))
    SUITE["operators"]["addition_dense_dense"][string(N)] = @benchmarkable ($op1 + $op2)
end

# Dense-dense multiplication
SUITE["operators"]["multiplication_dense_dense"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op1 = DenseOperator(b, rand(ComplexF64, N, N))
    op2 = DenseOperator(b, rand(ComplexF64, N, N))
    SUITE["operators"]["multiplication_dense_dense"][string(N)] = @benchmarkable ($op1 * $op2)
end

# Dense-ket multiplication
SUITE["operators"]["multiplication_dense_ket"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op = DenseOperator(b, rand(ComplexF64, N, N))
    psi = Ket(b, rand(ComplexF64, N))
    SUITE["operators"]["multiplication_dense_ket"][string(N)] = @benchmarkable ($op * $psi)
end

# Sparse-ket multiplication
SUITE["operators"]["multiplication_sparse_ket"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op = destroy(b)
    psi = Ket(b, rand(ComplexF64, N))
    SUITE["operators"]["multiplication_sparse_ket"][string(N)] = @benchmarkable ($op * $psi)
end

# =============================================================================
# State creation benchmarks
# =============================================================================

SUITE["states"] = BenchmarkGroup(["states"])

SUITE["states"]["coherentstate"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    alpha = log(N)
    SUITE["states"]["coherentstate"][string(N)] = @benchmarkable coherentstate($b, $alpha)
end

# =============================================================================
# Expectation value benchmarks
# =============================================================================

SUITE["expect"] = BenchmarkGroup(["expect"])

SUITE["expect"]["operator"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op = number(b)
    psi = coherentstate(b, 3.0)
    SUITE["expect"]["operator"][string(N)] = @benchmarkable expect($op, $psi)
end

SUITE["expect"]["variance"] = BenchmarkGroup()
for N in [50, 100, 200, 500]
    b = FockBasis(N-1)
    op = number(b)
    psi = coherentstate(b, 3.0)
    SUITE["expect"]["variance"][string(N)] = @benchmarkable variance($op, $psi)
end

# =============================================================================
# Partial trace benchmarks
# =============================================================================

SUITE["ptrace"] = BenchmarkGroup(["ptrace"])

SUITE["ptrace"]["state"] = BenchmarkGroup()
for N in [10, 20, 50]
    b1 = FockBasis(N-1)
    b2 = FockBasis(N-1)
    psi = tensor(coherentstate(b1, 2.0), coherentstate(b2, 1.0))
    SUITE["ptrace"]["state"][string(N)] = @benchmarkable ptrace($psi, 1)
end

SUITE["ptrace"]["operator"] = BenchmarkGroup()
for N in [10, 20, 50]
    b1 = FockBasis(N-1)
    b2 = FockBasis(N-1)
    rho = dm(tensor(coherentstate(b1, 2.0), coherentstate(b2, 1.0)))
    SUITE["ptrace"]["operator"][string(N)] = @benchmarkable ptrace($rho, 1)
end

# =============================================================================
# Phase space benchmarks
# =============================================================================

SUITE["phasespace"] = BenchmarkGroup(["phasespace"])

SUITE["phasespace"]["qfunc_state"] = BenchmarkGroup()
for N in [50, 100, 200]
    b = FockBasis(N-1)
    psi = coherentstate(b, 3.0)
    xvec = range(-5, 5, length=50)
    yvec = range(-5, 5, length=50)
    SUITE["phasespace"]["qfunc_state"][string(N)] = @benchmarkable qfunc($psi, $xvec, $yvec)
end

SUITE["phasespace"]["wigner_state"] = BenchmarkGroup()
for N in [50, 100, 200]
    b = FockBasis(N-1)
    psi = coherentstate(b, 3.0)
    xvec = range(-5, 5, length=50)
    yvec = range(-5, 5, length=50)
    SUITE["phasespace"]["wigner_state"][string(N)] = @benchmarkable wigner($psi, $xvec, $yvec)
end

# =============================================================================
# Time evolution with physics benchmarks (cavity, Jaynes-Cummings)
# =============================================================================

SUITE["timeevolution_physics"] = BenchmarkGroup(["timeevolution_physics"])

# Cavity Schrödinger evolution
SUITE["timeevolution_physics"]["schroedinger_cavity"] = BenchmarkGroup()
for N in [50, 100, 200]
    b = FockBasis(N-1)
    a = destroy(b)
    at = create(b)
    n = number(b)
    η = 1.5; ωc = 1.8; ωl = 2.0
    Δc = ωl - ωc
    H = Δc*at*a + η*(a + at)
    alpha0 = 0.3 - 0.5im
    psi0 = coherentstate(b, alpha0)
    tspan = range(0, 10, length=11)
    SUITE["timeevolution_physics"]["schroedinger_cavity"][string(N)] = @benchmarkable timeevolution.schroedinger($tspan, $psi0, $H; reltol=1e-6, abstol=1e-8) evals=1
end

# Jaynes-Cummings master equation
SUITE["timeevolution_physics"]["master_jaynescummings"] = BenchmarkGroup()
for N in [10, 20, 50]
    b_fock = FockBasis(N-1)
    b_spin = SpinBasis(1//2)
    a = destroy(b_fock) ⊗ one(b_spin)
    at = create(b_fock) ⊗ one(b_spin)
    sm = one(b_fock) ⊗ sigmam(b_spin)
    sp = one(b_fock) ⊗ sigmap(b_spin)
    ωc = 1.0; ωa = 1.0; g = 0.5; κ = 0.1; γ = 0.05
    H = ωc*at*a + ωa*sp*sm + g*(at*sm + a*sp)
    J = [sqrt(κ)*a, sqrt(γ)*sm]
    psi0 = tensor(fockstate(b_fock, 1), spinup(b_spin))
    tspan = range(0, 10, length=11)
    SUITE["timeevolution_physics"]["master_jaynescummings"][string(N)] = @benchmarkable timeevolution.master($tspan, $psi0, $H, $J; reltol=1e-6, abstol=1e-8) evals=1
end

# MCWF cavity
SUITE["timeevolution_physics"]["mcwf_cavity"] = BenchmarkGroup()
for N in [50, 100, 200]
    b = FockBasis(N-1)
    a = destroy(b)
    at = create(b)
    η = 1.5; ωc = 1.8; ωl = 2.0; κ = 0.5
    Δc = ωl - ωc
    H = Δc*at*a + η*(a + at)
    J = [sqrt(κ)*a]
    alpha0 = 0.3 - 0.5im
    psi0 = coherentstate(b, alpha0)
    tspan = range(0, 10, length=11)
    SUITE["timeevolution_physics"]["mcwf_cavity"][string(N)] = @benchmarkable timeevolution.mcwf($tspan, $psi0, $H, $J; seed=UInt(42), reltol=1e-6, abstol=1e-8) evals=1
end
