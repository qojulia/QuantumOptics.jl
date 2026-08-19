using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "ptrace_state"

samples = 5
evals = 200
cutoffs = [2:18;]

function setup(N)
    function create_substate(c0, alpha, N)
        b = GenericBasis(N)
        data = c0 .+ alpha*complex(range(0., stop=1., length=N))
        Ket(b, data)
    end
    psi1 = create_substate(1, 0.2, N)
    psi2 = create_substate(-2, 0.3, N)
    psi3 = create_substate(3, 0.4, 2)
    psi4 = create_substate(4, 0.5, 2)
    psi = tensor(psi1, psi2, psi3, psi4)
    psi
end

function f(psi)
    ptrace(psi, [2, 3])
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
    psi = setup(N)
    checks[N] = sum(abs.(f(psi).data))
    t = @belapsed f($psi) samples=samples evals=evals
    push!(results, Dict("N"=>4*N^2, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
