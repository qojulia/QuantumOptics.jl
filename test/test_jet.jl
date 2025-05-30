@testitem "JET Package Test" tags = [:jet] begin
    using JET
    using QuantumOptics

    JET.test_package(QuantumOptics, target_defined_modules = true)
end