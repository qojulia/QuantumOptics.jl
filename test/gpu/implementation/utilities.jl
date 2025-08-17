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

function create_multilevel_system(n, AT)
    """Create a multi-level system for more complex testing."""
    basis = NLevelBasis(n)
    
    # Create random Hamiltonian (Hermitian)
    H_data = rand(ComplexF64, n, n)
    H_data = H_data + H_data'  # Make Hermitian
    H = DenseOperator(basis, H_data)
    
    # Create initial state
    psi0_data = rand(ComplexF64, n)
    normalize!(psi0_data)
    psi0 = Ket(basis, psi0_data)
    
    # Adapt to GPU
    gpu_H = Adapt.adapt(AT, H)
    gpu_psi0 = Adapt.adapt(AT, psi0)
    
    return H, gpu_H, psi0, gpu_psi0
end

function create_coupled_oscillators(N, AT)
    """Create coupled harmonic oscillators system."""
    Ncutoff = 2
    basis_single = FockBasis(Ncutoff)
    basis = tensor([basis_single for i=1:N]...)
    
    # Parameters
    ω = ones(N)  # All oscillators have same frequency
    coupling = 0.1  # Weak coupling
    
    # Create Hamiltonian
    a = destroy(basis_single)
    at = create(basis_single)
    I = identityoperator(basis_single)
    
    # Individual oscillator terms
    H_terms = [embed(basis, i, ω[i] * at * a) for i=1:N]
    
    # Coupling terms
    for i=1:N-1
        H_coupling = embed(basis, [i, i+1], [a, coupling * at])
        push!(H_terms, H_coupling + dagger(H_coupling))
    end
    
    H = sum(H_terms)
    H = dense(H)  # Use dense operators for GPU testing
    
    # Initial state - coherent state in first mode
    psi0_terms = [coherentstate(basis_single, i == 1 ? 0.5 : 0.0) for i=1:N]
    psi0 = tensor(psi0_terms...)
    
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