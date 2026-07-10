using QuantumOptics
using BenchmarkTools
using SparseArrays
include("benchmarkutils.jl")

using Random; Random.seed!(0)

basename = "addition_sparse_sparse"

samples = 2
evals = 5
cutoffs = [50:50:1000;]
S = [0.1, 0.01, 0.001]
Nrand = 5

function setup(N, s)
    op1 = sprand(ComplexF64, N, N, s)
    op2 = sprand(ComplexF64, N, N, s)
    op1, op2
end

function f(op1, op2)
    op1 + op2
end

for s in S
    name = basename * "_" * replace(string(s), "." => "")
    
if isdefined(Main, :RUN_CI)
    SUITE[name] = BenchmarkGroup()
    for N in (length(cutoffs) > 0 ? [cutoffs[1]] : [])
        setup_args = setup(N)
        # some setups return a tuple, some return a single value
        if setup_args isa Tuple
            SUITE[name][string(N)] = @benchmarkable f($(setup_args)...)
        else
            SUITE[name][string(N)] = @benchmarkable f($setup_args)
        end
    end
else
    println("Benchmarking: ", name)

    print("Cutoff: ")
    results = []
    for N in cutoffs
        print(N, " ")
        T = 0.
        for i=1:Nrand
            op1, op2 = setup(N, s)
            T += @belapsed f($op1, $op2) samples=samples evals=evals
        end
        push!(results, Dict("N"=>N, "t"=>T/Nrand))
    end
    println()
    benchmarkutils.save(name, results)
end

end
