function test_schroedinger_gpu(AT, synchronize)
    """Test Schrödinger time evolution on GPU arrays."""
    
    @testset "Schrödinger Time Evolution GPU Tests" begin
        
        # Test 1: Basic Schrödinger evolution with single oscillator
        @testset "Single Oscillator" begin
            for n in test_sizes
                H, gpu_H, psi0, gpu_psi0 = create_test_system(n, AT)
                
                @test typeof(gpu_H.data) <: AT
                @test_broken typeof(gpu_psi0.data) <: AT
                
                # Run time evolution on CPU and GPU
                t_cpu, psi_cpu = timeevolution.schroedinger(T_SHORT, psi0, H)
                t_gpu, psi_gpu = timeevolution.schroedinger(T_SHORT, gpu_psi0, gpu_H)
                synchronize()
                
                # Verify results match
                @test_broken verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
                
                # Test state properties
                test_state_properties(psi_cpu, psi_gpu, synchronize)
            end
        end
    end
end