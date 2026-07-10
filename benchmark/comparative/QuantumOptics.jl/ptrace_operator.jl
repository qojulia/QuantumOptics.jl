using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "ptrace_operator"

samples = 5
evals = 200
cutoffs = [2:18;]

function setup(N)
    function create_suboperator(c0, alpha, N)
        b = GenericBasis(N)
        data = transpose(reshape(c0 .+ alpha.*complex(range(0., stop=1., length=N^2)), N, N))
        DenseOperator(b, data)
    end
    op1 = create_suboperator(1, 0.2, N)
    op2 = create_suboperator(-2, 0.3, N)
    op3 = create_suboperator(3, 0.4, 2)
    op4 = create_suboperator(4, 0.5, 2)
    op = tensor(op1, op2, op3, op4)
    op
end

function f(op)
    ptrace(op, [2, 3])
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
    op = setup(N)
    checks[N] = sum(abs.(f(op).data))
    t = @belapsed f($op) samples=samples evals=evals
    push!(results, Dict("N"=>4*N^2, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
