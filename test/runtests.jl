names = [
    "test_sortedindices.jl",
    "test_polynomials.jl",

    "test_bases.jl",
    "test_states.jl",

    "test_operators.jl",
    "test_operators_dense.jl",
    "test_sparsematrix.jl",
    "test_operators_sparse.jl",
    "test_operators_lazytensor.jl",
    "test_operators_lazysum.jl",
    "test_operators_lazyproduct.jl",

    "test_fock.jl",
    "test_spin.jl",
    "test_particle.jl",
    "test_manybody.jl",
    "test_nlevel.jl",
    "test_subspace.jl",
    "test_state_definitions.jl",

    "test_phasespace.jl",
    "test_transformations.jl",

    "test_metrics.jl",
    "test_embed.jl",
    "test_spectralanalysis.jl",

    "test_timeevolution_schroedinger.jl",
    "test_timeevolution_master.jl",
    "test_timeevolution_mcwf.jl",
    "test_timeevolution_bloch_redfield.jl",

    "test_timeevolution_twolevel.jl",
    "test_timeevolution_pumpedcavity.jl",

    "test_superoperators.jl",
    "test_steadystate.jl",
    "test_timecorrelations.jl",

    "test_semiclassical.jl",

    "test_stochastic_definitions.jl",
    "test_stochastic_schroedinger.jl",
    "test_stochastic_master.jl",
    "test_stochastic_semiclassical.jl",

    "test_printing.jl"
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
