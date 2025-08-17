# Utility functions for GPU time evolution testing

function create_test_system(n, AT)
    """Create a test quantum system and adapt it to the specified array type."""
    # Create basis
    basis = FockBasis(n-1)
    
    # Create operators
    a = destroy(basis)
    at = create(basis)
    H = dense(at * a)  # Number operator Hamiltonian
    
    # Create initial state
    psi0 = coherentstate(basis, 0.5)
    
    # Adapt to GPU
    gpu_H = Adapt.adapt(AT, H)
    gpu_psi0 = Adapt.adapt(AT, psi0)
    
    return H, gpu_H, psi0, gpu_psi0
end

function verify_timeevolution_result(cpu_result, gpu_result, tolerance=GPU_TOL)
    """Verify that GPU time evolution matches CPU result within tolerance."""
    if length(cpu_result[1]) != length(gpu_result[1])
        return false
    end
    
    # Check time vectors
    if !isapprox(cpu_result[1], gpu_result[1], atol=tolerance)
        return false
    end
    
    # Check state vectors
    for (cpu_state, gpu_state) in zip(cpu_result[2], gpu_result[2])
        gpu_data_cpu = Array(gpu_state.data)
        if !isapprox(cpu_state.data, gpu_data_cpu, atol=tolerance)
            return false
        end
    end
    
    return true
end

function test_state_properties(cpu_states, gpu_states, synchronize)
    """Test that GPU states maintain proper quantum properties."""
    @testset "State Properties" begin
        for (cpu_state, gpu_state) in zip(cpu_states, gpu_states)
            synchronize()
            
            # Test normalization
            cpu_norm = norm(cpu_state)
            gpu_norm = norm(gpu_state)
            synchronize()
            @test isapprox(cpu_norm, gpu_norm, atol=GPU_TOL)
            
            # Test that norm is approximately 1
            @test isapprox(gpu_norm, 1.0, atol=GPU_TOL)
        end
    end
end