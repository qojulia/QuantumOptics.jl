using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "variance_state"

samples = 5
evals = 100
cutoffs = [5000:5000:100000;]

function setup(N)
    b = FockBasis(N-1)
    op = (destroy(b) + create(b))
    psi = Ket(b, ones(ComplexF64, N)/sqrt(N))
    op, psi
end

function f(op, psi)
    variance(op, psi)
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
    op, psi = setup(N)
    checks[N] = abs(f(op, psi))
    t = @belapsed f($op, $psi) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
