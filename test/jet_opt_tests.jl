@testitem "JET optimization regressions" tags = [:jet] begin
using JET
using QuantumOptics
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
end
