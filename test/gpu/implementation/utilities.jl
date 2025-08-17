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

"""Verify that GPU time evolution matches CPU result within tolerance."""
function verify_timeevolution_result(cpu_result, gpu_result, tolerance=GPU_TOL)
    if length(cpu_result[1]) != length(gpu_result[1])
        return false
    end
    
    cpu_state = cpu_result[2][end]
    gpu_state = gpu_result[2][end]
    gpu_data_cpu = Array(gpu_state.data)
    return isapprox(cpu_state.data, gpu_data_cpu, atol=tolerance)
end