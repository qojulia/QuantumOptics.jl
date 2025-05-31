using TestItemRunner
using QuantumOptics

testfilter = ti->(!(:slow in ti.tags))
@run_package_tests filter=testfilter