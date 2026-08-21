@testitem "JET optimization regressions" tags = [:jet] begin
using JET
using QuantumOptics
using Random
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
end
