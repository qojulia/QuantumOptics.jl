@testitem "test_jet" tags = [:jet] begin
using Test
using QuantumOptics

@testset "JET Package Test" begin
    using JET

    JET.test_package(QuantumOptics, target_defined_modules = true)
end
end