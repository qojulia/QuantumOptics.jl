using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "coherentstate"

samples = 5
evals = 10000
cutoffs = [50:50:500;]

function setup(N)
    b = FockBasis(N-1)
    alpha = log(N)
    b, alpha
end

function f(b, alpha)
    coherentstate(b, alpha)
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    b, alpha = setup(N)
    checks[N] = abs(expect(destroy(b), f(b, alpha)))
    t = @belapsed f($b, $alpha) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
