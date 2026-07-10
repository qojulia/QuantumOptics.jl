using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "timeevolution_mcwf_jaynescummings"

samples = 2
evals = 100
cutoffs = [50:50:500;]
Ncheck = 200

function setup(N)
    nothing
end

function f(N)
    ωa = 1
    ωc = 1
    g = 2
    κ = 0.5
    γ = 0.1
    n_th = 0.75
    tspan = [0.0:1.0:10.0;]

    b_cavity = FockBasis(N-1)
    b_atom = SpinBasis(1//2)
    b = b_cavity ⊗ b_atom

    a = destroy(b_cavity)
    at = create(b_cavity)
    n = number(b_cavity)

    sz = sigmaz(b_atom)
    sm = sigmam(b_atom)
    sp = sigmap(b_atom)

    H = embed(b, 1, ωc*n) + embed(b, 2, ωa*sz/2) + g*(at⊗sm+a⊗sp)
    J = [sqrt(κ*(1+n_th))*embed(b, 1, a),
         sqrt(κ*n_th)*embed(b, 1, at),
         sqrt(γ)*embed(b, 2, sm)]
    ψ₀ = fockstate(b_cavity, 0) ⊗ normalize(spinup(b_atom) + spindown(b_atom))

    exp_n = Float64[]
    fout(t, ψ) = push!(exp_n, real(expect(1, n, ψ)/norm(ψ)^2))
    timeevolution.mcwf(tspan, ψ₀, H, J; fout=fout, reltol=1e-6, abstol=1e-8)
    exp_n
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
    C = 0.
    for i=1:Ncheck
        C += sum(f(N))
    end
    checks[N] = C/Ncheck
    t = @belapsed f($N) samples=samples evals=evals
    push!(results, Dict("N"=>2*N, "t"=>t))
end
println()

benchmarkutils.check(name, checks, 0.05)
benchmarkutils.save(name, results)

end
