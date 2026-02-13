using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "coherentstate"

samples = 5
evals = 10000
cutoffs = [50:50:500;]

function setup(N)
    alpha = log(N)
    alpha
end

function f(N, alpha)
    coherent(N, alpha)
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    alpha = setup(N)
    checks[N] = abs(expect(destroy(N), f(N, alpha)))
    t = @belapsed f($N, $alpha) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
