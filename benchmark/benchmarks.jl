using BenchmarkTools
using QuantumOptics
using OrdinaryDiffEq
using StochasticDiffEq
using LinearAlgebra

const SUITE = BenchmarkGroup()

prob_list = ("schrod", "master")
for prob in prob_list
    SUITE[prob] = BenchmarkGroup([prob])
    for type in ("pure", "custom")
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
    if pure
        obj = rho0.data
        Jobj, Jdag = (J.data, dagger(J).data)
        Hobj = H.data
    else
        obj = rho0
        Jobj, Jdag = (J, dagger(J))
        Hobj = H
    end
    master!(drho, rho, p, t) = timeevolution.dmaster_h!(drho, Hobj, [Jobj], [Jdag], nothing, rho, copy(obj))
    prob = ODEProblem(master!, obj, (t₀, t₁))
end

for dim in (1//2, 10//1, 50//1, 100//1)
    for prob in zip(prob_list, (:bench_schroedinger, :bench_master))
        name, bench = (prob[1], prob[2])
        # benchmark solving ODE problems on data of QO types
        SUITE[name]["pure"][string(dim)] = @benchmarkable solve(eval($bench)($dim; pure=true), DP5(); save_everystep=false)
        # benchmark solving ODE problems on custom QO types
        SUITE[name]["custom"][string(dim)] = @benchmarkable solve(eval($bench)($dim; pure=false), DP5(); save_everystep=false)
    end
end