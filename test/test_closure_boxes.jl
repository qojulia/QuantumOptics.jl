@testitem "test_closure_boxes" begin
using Test
using QuantumOptics

if VERSION >= v"1.14"
    @test isempty(Test.detect_closure_boxes(QuantumOptics))
end
end
