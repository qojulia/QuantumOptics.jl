using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "qfunc_state"

samples = 3
evals = 10
cutoffs = [50:50:500;]

function setup(N)
    psi = coherent(N, 3.0)
    xvec = range(-5, 5, length=50)
    yvec = range(-5, 5, length=50)
    psi, xvec, yvec
end

# Note: QuantumToolbox.jl uses a single xvec and computes the grid internally
function f(psi, xvec, yvec)
    wigner(psi, xvec)
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    psi, xvec, yvec = setup(N)
    checks[N] = sum(f(psi, xvec, yvec))
    t = @belapsed f($psi, $xvec, $yvec) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
