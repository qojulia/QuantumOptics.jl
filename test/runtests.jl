names = [
    "test_phasespace.jl",
    "test_spectralanalysis.jl",

    "test_timeevolution_schroedinger.jl",
    "test_timeevolution_master.jl",
    "test_timeevolution_mcwf.jl",
    "test_timeevolution_bloch_redfield.jl",
    "test_timeevolution_tdops.jl",

    "test_timeevolution_twolevel.jl",
    "test_timeevolution_pumpedcavity.jl",

    "test_steadystate.jl",
    "test_timecorrelations.jl",

    "test_semiclassical.jl",

    "test_stochastic_definitions.jl",
    "test_stochastic_schroedinger.jl",
    "test_stochastic_master.jl",
    "test_stochastic_semiclassical.jl",

    "test_timeevolution_abstractdata.jl",

    "test_sciml_broadcast_interfaces.jl",
    "test_ForwardDiff.jl"
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
