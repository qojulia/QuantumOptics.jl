# Generated from the cold-start scenarios on Julia 1.12.6 (x86_64-linux-gnu).
# Each raw trace was filtered with sortprecompile.py revision
# 4ea3c813d9a5adc494d84368e08825f5d765a0a6 using `-m 50.000001`.
# Timing comments preserve the largest observation for each exact statement.
# Recompilations and signatures tied to scenarios, the benchmark driver,
# package artifacts, generated names, private Base methods, compiler or solver
# internals, or modules not bound in QuantumOptics are excluded. These
# observations select statements; they are not benchmarks.

using PrecompileTools: @setup_workload

@setup_workload begin
    #=   51.9 ms =# precompile(Tuple{typeof(Core.kwcall), NamedTuple{(:alg, :by), Tuple{Base.Sort.QuickSortAlg, typeof(LinearAlgebra.eigsortby)}}, typeof(Base.sortperm), Array{Base.Complex{Float64}, 1}})
end
