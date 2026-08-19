using PrecompileTools

@setup_workload let
    # Closed-system, open-system, and quantum-trajectory time evolution
    basis = FockBasis(4)
    hamiltonian = number(basis)
    jumps = [destroy(basis)]
    initial_state = fockstate(basis, 1)
    times = [0.0, 0.1]

    @compile_workload begin
        schroedinger_times, pure_states =
            timeevolution.schroedinger(times, initial_state, hamiltonian)
        @assert schroedinger_times == times
        @assert isapprox(norm(pure_states[end]), 1; atol=1e-9)
        @assert isfinite(real(expect(hamiltonian, pure_states[end])))

        master_times, density_states =
            timeevolution.master(times, initial_state, hamiltonian, jumps)
        @assert master_times == times
        @assert isapprox(tr(density_states[end]), 1; atol=1e-10)
        @assert isfinite(real(expect(hamiltonian, density_states[end])))

        mcwf_times, trajectory_states = timeevolution.mcwf(
            times,
            initial_state,
            hamiltonian,
            jumps;
            seed=UInt(0),
        )
        @assert mcwf_times == times
        @assert isapprox(norm(trajectory_states[end]), 1; atol=1e-10)
        @assert isfinite(real(expect(hamiltonian, trajectory_states[end])))
    end
end
