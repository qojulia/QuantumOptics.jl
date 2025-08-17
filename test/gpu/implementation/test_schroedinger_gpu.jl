function test_schroedinger_gpu(AT, synchronize)
    """Test Schrödinger time evolution on GPU arrays."""
    
    @testset "Schrödinger Time Evolution GPU Tests" begin
        
        # Test 1: Basic Schrödinger evolution with single oscillator
        @testset "Single Oscillator" begin
            for n in test_sizes
                H, gpu_H, psi0, gpu_psi0 = create_test_system(n, AT)
                
                @test typeof(gpu_H.data) <: AT
                @test typeof(gpu_psi0.data) <: AT
                
                # Run time evolution on CPU and GPU
                t_cpu, psi_cpu = timeevolution.schroedinger(T_SHORT, psi0, H)
                t_gpu, psi_gpu = timeevolution.schroedinger(T_SHORT, gpu_psi0, gpu_H)
                synchronize()
                
                # Verify results match
                @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
                
                # Test state properties
                test_state_properties(psi_cpu, psi_gpu, synchronize)
            end
        end
        
        # Test 2: Multi-level system
        @testset "Multi-Level System" begin
            for n in test_sizes
                H, gpu_H, psi0, gpu_psi0 = create_multilevel_system(n, AT)
                
                # Run time evolution
                t_cpu, psi_cpu = timeevolution.schroedinger(T_SHORT, psi0, H)
                t_gpu, psi_gpu = timeevolution.schroedinger(T_SHORT, gpu_psi0, gpu_H)
                synchronize()
                
                # Verify results
                @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
            end
        end
        
        # Test 3: Coupled oscillators (smaller system due to tensor product)
        @testset "Coupled Oscillators" begin
            N = 2  # Two coupled oscillators
            H, gpu_H, psi0, gpu_psi0 = create_coupled_oscillators(N, AT)
            
            # Run time evolution
            t_cpu, psi_cpu = timeevolution.schroedinger(T_SHORT, psi0, H)
            t_gpu, psi_gpu = timeevolution.schroedinger(T_SHORT, gpu_psi0, gpu_H)
            synchronize()
            
            # Verify results
            @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
        end
        
        # Test 4: Time-dependent Hamiltonian
        @testset "Time-Dependent Hamiltonian" begin
            n = test_sizes[1]  # Use smallest size for time-dependent case
            basis = NLevelBasis(n)
            
            # Static part
            H0_data = rand(ComplexF64, n, n)
            H0_data = H0_data + H0_data'
            H0 = DenseOperator(basis, H0_data)
            
            # Time-dependent part  
            H1_data = rand(ComplexF64, n, n)
            H1_data = H1_data + H1_data'
            H1 = DenseOperator(basis, H1_data)
            
            # Initial state
            psi0_data = rand(ComplexF64, n)
            normalize!(psi0_data)
            psi0 = Ket(basis, psi0_data)
            
            # Time-dependent function
            f(t, psi) = H0 + sin(t) * H1
            f_gpu(t, psi) = Adapt.adapt(AT, H0) + sin(t) * Adapt.adapt(AT, H1)
            
            # GPU versions
            gpu_psi0 = Adapt.adapt(AT, psi0)
            
            # Run time evolution
            t_cpu, psi_cpu = timeevolution.schroedinger_dynamic(T_SHORT, psi0, f)
            t_gpu, psi_gpu = timeevolution.schroedinger_dynamic(T_SHORT, gpu_psi0, f_gpu)
            synchronize()
            
            # Verify results
            @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
        end
        
        # Test 5: Longer time evolution
        @testset "Longer Time Evolution" begin
            n = test_sizes[1]  # Use smallest size for longer evolution
            H, gpu_H, psi0, gpu_psi0 = create_test_system(n, AT)
            
            # Run longer time evolution
            t_cpu, psi_cpu = timeevolution.schroedinger(T_MEDIUM, psi0, H)
            t_gpu, psi_gpu = timeevolution.schroedinger(T_MEDIUM, gpu_psi0, gpu_H)
            synchronize()
            
            # Verify results
            @test verify_timeevolution_result((t_cpu, psi_cpu), (t_gpu, psi_gpu))
            
            # Test conservation of probability
            for psi in psi_gpu
                @test isapprox(norm(psi), 1.0, atol=GPU_TOL)
            end
        end
        
        # Test 6: Energy conservation for Hermitian Hamiltonian
        @testset "Energy Conservation" begin
            n = test_sizes[1]
            H, gpu_H, psi0, gpu_psi0 = create_test_system(n, AT)
            
            # Run time evolution
            t_gpu, psi_gpu = timeevolution.schroedinger(T_MEDIUM, gpu_psi0, gpu_H)
            synchronize()
            
            # Check energy conservation
            energies = [real(expect(gpu_H, psi)) for psi in psi_gpu]
            synchronize()
            
            initial_energy = energies[1]
            for energy in energies
                @test isapprox(energy, initial_energy, atol=1e-8)
            end
        end
    end
end