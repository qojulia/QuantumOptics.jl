using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "expect_operator"

samples = 5
evals = 1000
cutoffs = [50:50:500;]

function setup(N)
    b = FockBasis(N-1)
    op = number(b)
    psi = coherentstate(b, 3.0)
    op, psi
end

function f(op, psi)
    expect(op, psi)
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    op, psi = setup(N)
    checks[N] = real(f(op, psi))
    t = @belapsed f($op, $psi) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
