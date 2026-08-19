using QuantumToolbox
using BenchmarkTools
include("../QuantumOptics.jl/benchmarkutils.jl")
using Random; Random.seed!(0)

name = "addition_dense_dense"
samples = 3
evals = 10
cutoffs = [50:50:800;]

function setup(N)
    op1 = Qobj(rand(ComplexF64, N, N))
    op2 = Qobj(rand(ComplexF64, N, N))
    op1, op2
end

function f(op1, op2)
    op1 + op2
end

println("Benchmarking: ", name)
print("Cutoff: ")
results = []
for N in cutoffs
    print(N, " ")
    op1, op2 = setup(N)
    t = @belapsed f($op1, $op2) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()
benchmarkutils.save(name, results)
