using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "timeevolution_mcwf_cavity"

samples = 3
evals = 1
cutoffs = [50:50:500;]

function setup(N)
    nothing
end

function f(N)
    η = 1.5
    ωc = 1.8
    ωl = 2.0
    κ = 0.5
    Δc = ωl - ωc
    α0 = 0.3 - 0.5im
    tspan = [0:1.0:10;]

    a = destroy(N)
    at = create(N)
    n = num(N)

    H = Δc*at*a + η*(a + at)
    c_ops = [sqrt(κ)*a]

    psi0 = coherent(N, α0)
    sol = mcsolve(H, psi0, tspan; c_ops=c_ops, e_ops=[n], ntraj=1, seed=42, reltol=1e-6, abstol=1e-8)
    real.(sol.expect[1,:])
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    checks[N] = abs(sum(f(N)))
    t = @belapsed f($N) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
