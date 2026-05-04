function test_schroedinger_gpu(AT, synchronize; time_eltype=Float64, solver_kwargs=(;), kwargs...)
    """Test Schrödinger time evolution on GPU arrays."""

    @testset "Schrödinger Time Evolution GPU Tests" begin
        t_short = time_eltype.(T_SHORT)

        # Test 1: Basic Schrödinger evolution with single oscillator
        @testset "Single Oscillator" begin
            for n in test_sizes
                H, gpu_H, psi0, gpu_psi0 = create_test_system(n, AT; kwargs...)
                
                @test typeof(gpu_H.data) <: AT
                @test typeof(gpu_psi0.data) <: AT

                # Run time evolution on CPU and GPU
                t_cpu, psi_cpu = timeevolution.schroedinger(t_short, psi0, H; solver_kwargs...)
                t_gpu, psi_gpu = timeevolution.schroedinger(t_short, gpu_psi0, gpu_H; solver_kwargs...)
                synchronize()
                
                # Verify results match
                @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
            end
        end
    end
end
