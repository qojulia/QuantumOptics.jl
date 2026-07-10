using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

using Random; Random.seed!(0)

name = "multiplication_dense_ket"

samples = 2
evals = 5
cutoffs = [50:50:500;]

function setup(N)
    b = GenericBasis(N)
    op1 = randoperator(b)
    psi = randstate(b)
    result = copy(psi)
    op1, psi, result
end

function f(op1, psi, result)
    QuantumOpticsBase.gemv!(ComplexF64(1., 0.), op1, psi, ComplexF64(0., 0.), result)
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
    op1, psi, result = setup(N)
    t = @belapsed f($op1, $psi, $result) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()
benchmarkutils.save(name, results)

end
