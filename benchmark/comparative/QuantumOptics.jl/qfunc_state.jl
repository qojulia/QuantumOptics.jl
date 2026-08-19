using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "qfunc_state"

samples = 3
evals = 5
cutoffs = [10:10:100;]

function setup(N)
    alpha = 0.7
    xvec = collect(range(-50, stop=50, length=100))
    yvec = collect(range(-50, stop=50, length=100))
    b = FockBasis(N-1)
    state = coherentstate(b, alpha)
    state, xvec, yvec
end

function f(state, xvec, yvec)
    qfunc(state, xvec, yvec)
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
    state, xvec, yvec = setup(N)
    alpha_check = 0.6 + 0.1im
    checks[N] = f(state, [real(alpha_check)], [imag(alpha_check)])[1, 1]
    t = @belapsed f($state, $xvec, $yvec) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
