using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

using Random; Random.seed!(0)

name = "multiplication_dense_dense"

samples = 3
evals = 10
cutoffs = [50:50:1000;]

function setup(N)
    b = GenericBasis(N)
    op1 = randoperator(b)
    op2 = randoperator(b)
    result = DenseOperator(b)
    op1, op2, result
end

function f(op1, op2, result)
    QuantumOpticsBase.gemm!(ComplexF64(1., 0.), op1, op2, ComplexF64(0., 0.), result)
end


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
    op1, op2, result = setup(N)
    t = @belapsed f($op1, $op2, $result) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.save(name, results)

end
