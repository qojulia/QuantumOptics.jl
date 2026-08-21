@testitem "JET optimization regressions" tags = [:jet] begin
using JET
using QuantumOptics
using LinearAlgebra
using Random
using SparseArrays
using Test

promoted_span_sum(args...) =
    sum(first(QuantumOptics.timeevolution._promote_time_and_state(args...)))

@testset "Time and state promotion" begin
    b = SpinBasis(1//2)
    psi = spindown(b)
    rho = dm(psi)
    H = sigmax(b)
    J = [sigmam(b)]
    times = [0.0, 0.1, 0.2]

    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
        @__MODULE__,
    ) promoted_span_sum(psi, H, times)

    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
        @__MODULE__,
    ) promoted_span_sum(rho, H, J, times)
end

@testset "Semiclassical broadcasting" begin
    b = SpinBasis(1//2)
    psi = spindown(b)
    classical = ComplexF64[0.7, 0.2]

    ket_state = semiclassical.State(psi, classical)
    operator_state = semiclassical.State(dm(psi), classical)

    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
    ) Base.materialize(Base.broadcasted(*, ket_state, 2.0))

    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
    ) Base.materialize(Base.broadcasted(*, operator_state, 2.0))
end

@testset "Mixed jump operator callbacks" begin
    b = SpinBasis(1//2)
    psi = spinup(b)
    rho = dm(psi)
    H = dense(0.3 * sigmax(b))
    sparse_jump = sigmam(b)
    J = [sparse_jump, dense(sparse_jump)]
    Jdagger = dagger.(J)
    rates = [0.4, 0.7]
    Hnh = timeevolution.nh_hamiltonian(H, J, Jdagger, rates)
    Hnhdagger = dagger(Hnh)

    drho = copy(rho)
    dmaster_h = timeevolution._dmaster_h_function(H, J, Jdagger, rates, copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_h(0.0, rho, drho)

    dmaster_nh = timeevolution._dmaster_nh_function(
        Hnh, Hnhdagger, J, Jdagger, rates, copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_nh(0.0, rho, drho)

    dpsi = copy(psi)
    dmcwf_h = timeevolution._dmcwf_h_function(H, J, Jdagger, rates, copy(psi))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmcwf_h(0.0, psi, dpsi)

    rng = Xoshiro(1)
    psi_new = copy(psi)
    jump_with_rates = timeevolution._jump_function(J, rates, zeros(length(J)))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) jump_with_rates(rng, 0.0, psi, psi_new)

    jump_without_rates = timeevolution._jump_function(J, nothing, zeros(length(J)))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) jump_without_rates(rng, 0.0, psi, psi_new)
end

@testset "Fundamental time-evolution callbacks" begin
    b = SpinBasis(1//2)
    psi = spinup(b)
    rho = dm(psi)
    H = dense(0.3 * sigmax(b))
    sparse_jump = sigmam(b)
    J = (sparse_jump, dense(sparse_jump))
    Jdagger = dagger.(J)
    rates = [0.4, 0.7]
    Hnh = timeevolution.nh_hamiltonian(H, J, Jdagger, rates)
    Hnhdagger = dagger(Hnh)

    dschroedinger = timeevolution._dschroedinger_function(H)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dschroedinger(0.0, psi, copy(psi))

    H_dynamic = TimeDependentSum(cos => H, sin => dense(sigmaz(b)))
    fschroedinger = timeevolution.schroedinger_dynamic_function(H_dynamic)
    dschroedinger_dynamic = timeevolution._dschroedinger_dynamic_function(fschroedinger)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dschroedinger_dynamic(0.0, psi, copy(psi))

    # The non-Hermitian MCWF solver uses the same prepared callback as
    # the static Schrödinger solver.
    dmcwf_nh = timeevolution._dschroedinger_function(Hnh)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmcwf_nh(0.0, psi, copy(psi))

    fmaster = let H = H, J = J, Jdagger = Jdagger
        (t, state) -> (H, J, Jdagger)
    end
    fmaster_nh = let Hnh = Hnh, Hnhdagger = Hnhdagger, J = J, Jdagger = Jdagger
        (t, state) -> (Hnh, Hnhdagger, J, Jdagger)
    end

    dmaster_dynamic = timeevolution._dmaster_h_dynamic_function(
        fmaster, rates, copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_dynamic(0.0, rho, copy(rho))

    dmaster_nh_dynamic = timeevolution._dmaster_nh_dynamic_function(
        fmaster_nh, rates, copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_nh_dynamic(0.0, rho, copy(rho))

    L = liouvillian(H, collect(J); rates=rates)
    vector_basis = GenericBasis(length(rho))
    rho_vector = Ket(vector_basis, vec(copy(rho.data)))
    L_vector = Operator(vector_basis, vector_basis, L.data)
    dmaster_liouville = timeevolution._dmaster_liouville_function(L_vector)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_liouville(0.0, rho_vector, copy(rho_vector))

    dmcwf_dynamic = timeevolution._dmcwf_h_dynamic_function(
        fmaster, rates, copy(psi))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmcwf_dynamic(0.0, psi, copy(psi))

    fmcwf_nh = let Hnh = Hnh, J = J, Jdagger = Jdagger
        (t, state) -> (Hnh, J, Jdagger)
    end
    dmcwf_nh_dynamic = timeevolution._dmcwf_nh_dynamic_function(fmcwf_nh)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmcwf_nh_dynamic(0.0, psi, copy(psi))

    jump_dynamic = timeevolution._jump_dynamic_function(
        fmaster, rates, zeros(length(J)))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) jump_dynamic(Xoshiro(1), 0.0, psi, copy(psi))

    ode_rhs = timeevolution._ode_rhs_function(
        dschroedinger, copy(psi), copy(psi))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) ode_rhs(similar(psi.data), copy(psi.data), nothing, 0.0)

    master_ode_rhs = timeevolution._ode_rhs_function(
        dmaster_dynamic, copy(rho), copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) master_ode_rhs(similar(rho.data), copy(rho.data), nothing, 0.0)

    bloch_redfield_basis = b^2
    L_bloch_redfield = SparseOperator(bloch_redfield_basis, sparse(L.data))
    rho_bloch_redfield = Ket(bloch_redfield_basis, vec(copy(rho.data)))
    dmaster_bloch_redfield = timeevolution._dmaster_br_function(L_bloch_redfield)
    JET.@test_opt target_modules = (
        QuantumOptics.timeevolution,
    ) dmaster_bloch_redfield(
        0.0, rho_bloch_redfield, copy(rho_bloch_redfield))
end

@testset "Semiclassical solver callbacks" begin
    b = SpinBasis(1//2)
    psi = spinup(b)
    rho = dm(psi)
    H = dense(0.3 * sigmax(b))
    J = (dense(sigmam(b)),)
    Jdagger = dagger.(J)
    rates = [0.4]
    classical = ComplexF64[0.2]
    ket_state = semiclassical.State(psi, copy(classical))
    rho_state = semiclassical.State(rho, copy(classical))

    fschroedinger = let H = H
        (t, state, classical) -> H
    end
    fmaster = let H = H, J = J, Jdagger = Jdagger
        (t, state, classical) -> (H, J, Jdagger)
    end
    fclassical! = (dclassical, classical, state, t) ->
        fill!(dclassical, zero(eltype(dclassical)))
    fjump_classical! = (classical, state, index, t) -> nothing

    dschroedinger = semiclassical._dschroedinger_dynamic_function(
        fschroedinger, fclassical!)
    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dschroedinger(0.0, ket_state, copy(ket_state))

    dmaster = semiclassical._dmaster_h_dynamic_function(
        fmaster, fclassical!, rates, copy(rho))
    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dmaster(0.0, rho_state, copy(rho_state))

    dmcwf = semiclassical._dmcwf_h_dynamic_function(
        fmaster, fclassical!, rates, copy(psi))
    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dmcwf(0.0, ket_state, copy(ket_state))

    jump = semiclassical._jump_dynamic_function(
        fmaster, fclassical!, fjump_classical!, rates, zeros(length(J)))
    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) jump(Xoshiro(1), 0.0, ket_state, copy(ket_state))

    x = zeros(ComplexF64, length(ket_state))
    semiclassical.recast!(x, ket_state)
    ode_rhs = timeevolution._ode_rhs_function(
        dschroedinger, copy(ket_state), copy(ket_state))
    JET.@test_opt target_modules = (
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) ode_rhs(similar(x), x, nothing, 0.0)
end

@testset "Stochastic solver callbacks" begin
    b = SpinBasis(1//2)
    psi = spinup(b)
    rho = dm(psi)
    H = dense(0.3 * sigmax(b))
    Hs = [0.1 * dense(sigmaz(b))]
    C = [0.1 * dense(sigmam(b))]
    Cdagger = dagger.(C)
    C_multiple = [C[1], 0.2 * H]
    Cdagger_multiple = dagger.(C_multiple)

    dschroedinger_stochastic = stochastic._dschroedinger_stochastic_function(Hs)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) dschroedinger_stochastic(
        zeros(ComplexF64, length(psi)), 0.0, psi, copy(psi), 1)

    homodyne_operators = [0.2 * dense(sigmam(b)), 0.1 * dense(sigmaz(b))]
    fdeterm_schroedinger, fstochastic_schroedinger =
        stochastic.homodyne_carmichael(H, homodyne_operators, [0.2, 0.3])
    dschroedinger_deterministic_dynamic =
        timeevolution._dschroedinger_dynamic_function(fdeterm_schroedinger)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) dschroedinger_deterministic_dynamic(0.0, psi, copy(psi))

    dschroedinger_stochastic_dynamic =
        stochastic._dschroedinger_stochastic_dynamic_function(
            fstochastic_schroedinger)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) dschroedinger_stochastic_dynamic(
        zeros(ComplexF64, length(psi), length(homodyne_operators)),
        0.0, psi, copy(psi), length(homodyne_operators))

    dmaster_stochastic = stochastic._dmaster_stochastic_function(C, Cdagger)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) dmaster_stochastic(
        zeros(ComplexF64, length(rho)), 0.0, rho, copy(rho), 1)

    fstochastic_master = let C = C_multiple, Cdagger = Cdagger_multiple
        (t, state) -> (C, Cdagger)
    end
    dmaster_stochastic_dynamic = stochastic._dmaster_stochastic_dynamic_function(
        fstochastic_master)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) dmaster_stochastic_dynamic(
        zeros(ComplexF64, length(rho), length(C_multiple)),
        0.0, rho, copy(rho), length(C_multiple))

    sde_noise = stochastic._sde_noise_function(
        dschroedinger_stochastic_dynamic,
        copy(psi), copy(psi), length(homodyne_operators))
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) sde_noise(
        zeros(ComplexF64, length(psi), length(homodyne_operators)),
        copy(psi.data), nothing, 0.0)

    x = stochastic.as_vector(rho)
    master_sde_noise = stochastic._sde_noise_function(
        dmaster_stochastic, copy(rho), copy(rho), 1)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.timeevolution,
    ) master_sde_noise(similar(x), copy(x), nothing, 0.0)
end

@testset "Stochastic semiclassical solver callbacks" begin
    b = SpinBasis(1//2)
    psi = spinup(b)
    rho = dm(psi)
    Hs = [0.1 * dense(sigmaz(b))]
    C = [0.1 * dense(sigmam(b))]
    Cdagger = dagger.(C)
    classical = ComplexF64[0.2]
    ket_state = semiclassical.State(psi, copy(classical))
    rho_state = semiclassical.State(rho, copy(classical))

    fstochastic_schroedinger = let Hs = Hs
        (t, state, classical) -> Hs
    end
    fstochastic_master = let C = C, Cdagger = Cdagger
        (t, state, classical) -> (C, Cdagger)
    end
    fstochastic_classical! = (dclassical, classical, state, t) ->
        fill!(dclassical, one(eltype(dclassical)))

    dschroedinger_quantum =
        stochastic._dschroedinger_semiclassical_stochastic_function(
            fstochastic_schroedinger, nothing)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dschroedinger_quantum(
        zeros(ComplexF64, length(ket_state)),
        0.0, ket_state, copy(ket_state), 1)

    dschroedinger_classical =
        stochastic._dschroedinger_semiclassical_stochastic_function(
            nothing, fstochastic_classical!)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dschroedinger_classical(
        zeros(ComplexF64, length(ket_state), 1),
        0.0, ket_state, copy(ket_state), 0)

    dschroedinger_combined =
        stochastic._dschroedinger_semiclassical_stochastic_function(
            fstochastic_schroedinger, fstochastic_classical!)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dschroedinger_combined(
        zeros(ComplexF64, length(ket_state), 3),
        0.0, ket_state, copy(ket_state), 1)

    dmaster_quantum = stochastic._dmaster_semiclassical_stochastic_function(
        fstochastic_master, nothing)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dmaster_quantum(
        zeros(ComplexF64, length(rho_state)),
        0.0, rho_state, copy(rho_state), 1)

    dmaster_classical = stochastic._dmaster_semiclassical_stochastic_function(
        nothing, fstochastic_classical!)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dmaster_classical(
        zeros(ComplexF64, length(rho_state), 1),
        0.0, rho_state, copy(rho_state), 0)

    dmaster_combined = stochastic._dmaster_semiclassical_stochastic_function(
        fstochastic_master, fstochastic_classical!)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) dmaster_combined(
        zeros(ComplexF64, length(rho_state), 3),
        0.0, rho_state, copy(rho_state), 1)

    sde_noise = stochastic._sde_noise_function(
        dmaster_combined, copy(rho_state), copy(rho_state), 1)
    x = zeros(ComplexF64, length(rho_state))
    semiclassical.recast!(x, rho_state)
    JET.@test_opt target_modules = (
        QuantumOptics.stochastic,
        QuantumOptics.semiclassical,
        QuantumOptics.timeevolution,
    ) sde_noise(
        zeros(ComplexF64, length(rho_state), 3),
        x, nothing, 0.0)
end

@testset "Steady-state solvers" begin
    b = SpinBasis(1//2)
    rho = dm(spinup(b))
    H = dense(0.3 * sigmax(b))
    sparse_jump = sigmam(b)
    dense_jump = dense(sparse_jump)

    L = liouvillian(H, [dense_jump])
    JET.@test_opt target_modules = (
        QuantumOptics.steadystate,
    ) steadystate.eigenvector(L)

    mixed_jumps = [sparse_jump, dense_jump]
    Jdagger = dagger.(mixed_jumps)
    rates = [0.4, 0.7]
    linear_map = steadystate._linmap_liouvillian(
        copy(rho), H, mixed_jumps, Jdagger, rates)
    x = [vec(rho.data); zero(eltype(rho))]
    y = similar(x)

    # The heterogeneous operator vector currently causes runtime dispatch in
    # the matrix-free iterative Liouvillian.
    JET.@test_opt broken = true target_modules = (
        QuantumOptics.steadystate,
        QuantumOptics.timeevolution,
    ) LinearAlgebra.mul!(y, linear_map, x)
end
end
