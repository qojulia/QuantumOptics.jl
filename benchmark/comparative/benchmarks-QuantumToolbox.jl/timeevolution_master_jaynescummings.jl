using QuantumToolbox
using BenchmarkTools
include("benchmarkutils.jl")

name = "timeevolution_master_jaynescummings"

samples = 3
evals = 1
cutoffs = [10:10:50;]

function setup(N)
    nothing
end

function f(N)
    ωc = 1.0
    ωa = 1.0
    g = 0.5
    κ = 0.1
    γ = 0.05
    tspan = [0:1.0:10;]

    a = kron(destroy(N), eye(2))
    sm = kron(eye(N), sigmam())
    sp = kron(eye(N), sigmap())
    n = a' * a

    H = ωc*a'*a + ωa*sp*sm + g*(a'*sm + a*sp)
    c_ops = [sqrt(κ)*a, sqrt(γ)*sm]

    psi0 = kron(fock(N, 1), fock(2, 0))
    sol = mesolve(H, psi0, tspan; c_ops=c_ops, e_ops=[n], reltol=1e-6, abstol=1e-8)
    real.(sol.expect[1,:])
end

println("Benchmarking: ", name)
print("Cutoff: ")
checks = Dict{Int, Float64}()
results = []
for N in cutoffs
    print(N, " ")
    checks[N] = abs(sum(f(N)))
    t = @belapsed f($N) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
