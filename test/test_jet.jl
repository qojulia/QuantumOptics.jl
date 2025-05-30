@testitem "test_jet" begin
using Test
using QuantumOptics

@testset "JET Package Test" tags = [:jet] begin
    using JET

    JET.test_package(QuantumOptics, target_defined_modules = true)
end
end