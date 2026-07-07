#!/usr/bin/env julia

# Lightweight benchmark smoke gate for CI and local development.
#
# This is intentionally not a full performance benchmark. Its job is to keep the
# benchmark project from bitrotting by checking that the benchmark environment
# instantiates, the benchmark suite loads, and at least one tiny benchmark can run.
#
# Local command from repository root:
#
#     julia --project=benchmark benchmark/run_smoke.jl
#
# Optional environment variables:
#
#     QO_BENCHMARK_SMOKE_GROUP=schroedinger
#     QO_BENCHMARK_SMOKE_KIND="base array types"
#     QO_BENCHMARK_SMOKE_SIZE="1//2"

using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

include(joinpath(@__DIR__, "benchmarks.jl"))

if !isdefined(Main, :SUITE)
    error("benchmark/benchmarks.jl did not define Main.SUITE")
end

using BenchmarkTools

const GROUP = get(ENV, "QO_BENCHMARK_SMOKE_GROUP", "schroedinger")
const KIND = get(ENV, "QO_BENCHMARK_SMOKE_KIND", "base array types")
const SIZE = get(ENV, "QO_BENCHMARK_SMOKE_SIZE", "1//2")

function _require_key(container, key, label)
    if !haskey(container, key)
        available = join(string.(collect(keys(container))), ", ")
        error("Missing benchmark $label key '$key'. Available: $available")
    end
    return container[key]
end

group_suite = _require_key(SUITE, GROUP, "group")
kind_suite = _require_key(group_suite, KIND, "kind")
bench = _require_key(kind_suite, SIZE, "size")

println("QuantumOptics benchmark smoke")
println("Julia version: ", VERSION)
println("Benchmark project: ", Base.active_project())
println("Selected benchmark: ", repr((GROUP, KIND, SIZE)))

trial = run(bench; samples=1, evals=1, seconds=1)
display(trial)
println()
println("BENCHMARK_SMOKE_OK")
