using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "wigner_state"

samples = 3
evals = 10
cutoffs = [50:50:500;]

function setup(N)
    psi = coherent(N, 3.0)
    xvec = range(-5, 5, length=50)
    psi, xvec
end

function f(psi, xvec)
    wigner(psi, xvec)
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    psi, xvec = setup(N)
    w, _, _ = f(psi, xvec)
    checks[N] = sum(w)
    t = @belapsed f($psi, $xvec) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
