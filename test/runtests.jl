names = [
    "test_sortedindices.jl",
    "test_bases.jl",
    "test_states.jl",

    "test_operators_dense.jl",
    "test_sparsematrix.jl",
    "test_operators_sparse.jl",
    "test_operators_lazytensor.jl",
    "test_operators_lazysum.jl",
    "test_operators_lazyproduct.jl",
    "test_identityoperator.jl",

    "test_fock.jl",
    "test_spin.jl",
    "test_particle.jl",
    "test_nparticles.jl",
    "test_nlevel.jl",
    "test_subspace.jl",

    "test_metrics.jl",
    "test_embed.jl",
    "test_spectralanalysis.jl",

    "test_odedopri.jl",
    "test_timeevolution_schroedinger.jl",
    "test_timeevolution_master.jl",
    "test_timeevolution_mcwf.jl",

    "test_superoperators.jl",
    "test_steadystate.jl",
    "test_timecorrelations.jl",

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

unavailable_tests = setdiff(names, detected_tests)
if length(unavailable_tests) != 0
    error("The following tests could not be found:\n", join(unavailable_tests, "\n"))
end

for name=names
    if startswith(name, "test_") && endswith(name, ".jl")
        include(name)
    end
end
