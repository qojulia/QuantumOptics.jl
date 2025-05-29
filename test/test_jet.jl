using QuantumOptics

@testitem "JET Package Test" tags = [:jet] begin
    using JET

    JET.test_package(QuantumOptics, target_defined_modules = true)
end