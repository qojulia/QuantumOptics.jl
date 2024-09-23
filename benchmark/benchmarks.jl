using BenchmarkTools
using QuantumOptics
using OrdinaryDiffEq
using StochasticDiffEq
using LinearAlgebra
using PkgBenchmark


const SUITE = BenchmarkGroup()

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
        # benchmark solving ODE problems on data of QO types
        SUITE[name]["base array types"][string(dim)] = @benchmarkable solve(prob, DP5(); save_everystep=false) setup=(prob=eval($bench)($dim; pure=true))
        # benchmark solving ODE problems on custom QO types
        SUITE[name]["qo types"][string(dim)] = @benchmarkable solve(prob, DP5(); save_everystep=false) setup=(prob=eval($bench)($dim; pure=false))
    end
    for solver in zip(("stochastic schroedinger", "stochastic master"), (:(bench_stochastic_schroedinger), :(bench_stochastic_master)))
        name, bench = (solver[1], solver[2])
        # benchmark solving ODE problems on data of QO types
        SUITE[name]["base array types"][string(dim)] = @benchmarkable solve(prob, EM(), dt=1/100; save_everystep=false) setup=(prob=eval($bench)($dim; pure=true))
        # benchmark solving ODE problems on custom QO types
        SUITE[name]["qo types"][string(dim)] = @benchmarkable solve(prob, EM(), dt=1/100; save_everystep=false) setup=(prob=eval($bench)($dim; pure=false))
    end
end