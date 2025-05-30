using TestItemRunner
using QuantumOptics

testfilter = ti->(startswith(ti.filename, "test_") && endswith(ti.filename, ".jl") )
@run_package_tests filter=testfilter