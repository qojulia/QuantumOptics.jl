using Test
using QuantumOptics

@testset "Quality Assurance" begin
    using Aqua

    Aqua.test_all(QuantumOptics, piracies = false, unbound_args = false)
end