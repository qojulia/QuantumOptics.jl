# Platform-agnostic GPU test runner
include("definitions.jl")
include("imports.jl") 
include("utilities.jl")
include("test_schroedinger_gpu.jl")

function test_platform(AT, synchronize)
    """Run all GPU tests for the specified array type."""
    
    @testset "QuantumOptics GPU Tests - $(AT)" begin
        # Test basic Schrödinger time evolution
        test_schroedinger_gpu(AT, synchronize)
    end
end