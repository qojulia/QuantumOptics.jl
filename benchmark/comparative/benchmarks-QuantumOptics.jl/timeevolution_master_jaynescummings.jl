using QuantumOptics
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
    tspan = [0:1.:10;]

    b_fock = FockBasis(N-1)
    b_spin = SpinBasis(1//2)

    a = destroy(b_fock) ⊗ one(b_spin)
    at = create(b_fock) ⊗ one(b_spin)
    sm = one(b_fock) ⊗ sigmam(b_spin)
    sp = one(b_fock) ⊗ sigmap(b_spin)

    H = ωc*at*a + ωa*sp*sm + g*(at*sm + a*sp)
    J = [sqrt(κ)*a, sqrt(γ)*sm]

    psi0 = tensor(fockstate(b_fock, 1), spinup(b_spin))
    n = at*a
    exp_n = Float64[]
    fout(t, ρ) = push!(exp_n, real(expect(n, ρ)))
    timeevolution.master(tspan, psi0, H, J; fout=fout, reltol=1e-6, abstol=1e-8)
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
