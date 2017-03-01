names = [
    "test_bases.jl",
    "test_states.jl",

    "test_operators.jl",
    "test_sparsematrix.jl",
    "test_sparseoperators.jl",
    "test_lazyoperators.jl",
    "test_lazytensor.jl",
    "test_lazysum.jl",
    "test_lazyproduct.jl",
    "test_identityoperator.jl",
    "test_operatorarithmetic.jl",

    "test_fock.jl",
    "test_spin.jl",
    "test_particle.jl",
    "test_nparticles.jl",
    "test_nlevel.jl",
    "test_subspace.jl",

    "test_metrics.jl",
    "test_tensorproduct.jl",
    "test_ptrace.jl",
    "test_permutesystems.jl",
    "test_embed.jl",
    "test_spectralanalysis.jl",

    "test_odedopri.jl",
    "test_timeevolution_schroedinger.jl",
    "test_timeevolution_master.jl",
    "test_timeevolution_mcwf.jl",

    "test_superoperators.jl",
    "test_steadystate.jl",
    "test_correlations.jl",

    "test_cumulantexpansion.jl",
    "test_cumulantexpansion_timedependent.jl",
    "test_correlationexpansion.jl",
    "test_correlationexpansion_mpc.jl",
]

detected_tests = filter(
    name->startswith(name, "test_") && endswith(name, ".jl"),
    readdir("."))
unused_tests = setdiff(detected_tests, names)

if length(unused_tests) != 0
    error("The following tests are not used:\n", join(unused_tests, "\n"))
end

for name=names
    if startswith(name, "test_") && endswith(name, ".jl")
        include(name)
    end
end
