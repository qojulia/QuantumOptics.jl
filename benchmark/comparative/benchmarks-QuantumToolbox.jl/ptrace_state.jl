using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "ptrace_state"

samples = 5
evals = 100
cutoffs = [10:10:50;]

function setup(N)
    psi = kron(coherent(N, 2.0), coherent(N, 1.0))
    psi
end

function f(psi)
    ptrace(psi, [1])
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    psi = setup(N)
    rho = f(psi)
    checks[N] = real(tr(rho))
    t = @belapsed f($psi) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
