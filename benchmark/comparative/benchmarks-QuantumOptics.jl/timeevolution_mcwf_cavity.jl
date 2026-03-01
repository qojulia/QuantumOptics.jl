using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "timeevolution_mcwf_cavity"

samples = 3
evals = 1
cutoffs = [50:50:500;]

function setup(N)
    nothing
end

function f(N)
    η = 1.5
    ωc = 1.8
    ωl = 2.
    κ = 0.5
    Δc = ωl - ωc
    α0 = 0.3 - 0.5im
    tspan = [0:1.:10;]

    b = FockBasis(N-1)
    a = destroy(b)
    at = create(b)
    n = number(b)

    H = Δc*at*a + η*(a + at)
    J = [sqrt(κ)*a]

    Ψ₀ = coherentstate(b, α0)
    exp_n = Float64[]
    fout(t, ρ) = push!(exp_n, real(expect(n, ρ)))
    timeevolution.mcwf(tspan, Ψ₀, H, J; fout=fout, seed=UInt(42), reltol=1e-6, abstol=1e-8)
    exp_n
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
