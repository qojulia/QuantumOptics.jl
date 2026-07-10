using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "displace"

samples = 3
evals = 10
cutoffs = [10:10:150;]

function setup(N)
    b = FockBasis(N-1)
    alpha = log(N)
    b, alpha
end

function f(b, alpha)
    displace(b, alpha)
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
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    b, alpha = setup(N)
    checks[N] = expect(destroy(b), f(b, alpha))
    t = @belapsed f($b, $alpha) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
