using QuantumOptics
using BenchmarkTools
include("benchmarkutils.jl")

name = "timeevolution_schroedinger_particle"

samples = 3
evals = 1
cutoffs = [25:25:300;]

function setup(N)
    xmin = -5
    xmax = 5
    x0 = 0.3
    p0 = -0.2
    sigma0 = 1
    bx = PositionBasis(xmin, xmax, N)
    x = position(bx)
    p = momentum(bx)
    H = p^2 + 2*x^2
    psi0 = gaussianstate(bx, x0, p0, sigma0)
    psi0, H, x
end

function f(psi0, H, x)
    T = [0:1.:10;]
    exp_x = Float64[]
    fout(t, psi) = push!(exp_x, real(expect(x, psi)))
    timeevolution.schroedinger(T, psi0, H; fout=fout, reltol=1e-6, abstol=1e-8)
    exp_x
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
    psi0, H, x = setup(N)
    checks[N] = sum(f(psi0, H, x))
    t = @belapsed f($psi0, $H, $x) samples=samples evals=evals
    push!(results, Dict("N"=>N, "t"=>t))
end
println()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)

end
