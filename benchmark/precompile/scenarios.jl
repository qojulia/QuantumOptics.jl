function schroedinger()
    basis = FockBasis(4)
    hamiltonian = number(basis)
    initial_state = fockstate(basis, 1)
    times = [0.0, 0.1]

    output_times, states =
        timeevolution.schroedinger(times, initial_state, hamiltonian)

    check(output_times == times, "Schrödinger evolution returned the wrong times")
    check(length(states) == length(times), "Schrödinger evolution returned the wrong state count")
    check(isapprox(norm(states[end]), 1; atol=1e-9), "Schrödinger evolution changed the state norm")
    return real(expect(hamiltonian, states[end]))
end

function master()
    basis = FockBasis(4)
    hamiltonian = number(basis)
    initial_state = fockstate(basis, 1)
    jump = destroy(basis)
    times = [0.0, 0.1]

    output_times, states =
        timeevolution.master(times, initial_state, hamiltonian, [jump])

    check(output_times == times, "master evolution returned the wrong times")
    check(length(states) == length(times), "master evolution returned the wrong state count")
    check(isapprox(tr(states[end]), 1; atol=1e-10), "master evolution is not trace preserving")
    check(isfinite(norm(states[end])), "master evolution returned a non-finite state")
    return real(expect(hamiltonian, states[end]))
end

function mcwf()
    basis = FockBasis(4)
    hamiltonian = number(basis)
    initial_state = fockstate(basis, 1)
    jump = destroy(basis)
    times = [0.0, 0.1]

    output_times, states = timeevolution.mcwf(
        times,
        initial_state,
        hamiltonian,
        [jump];
        seed=UInt(0),
    )

    check(output_times == times, "MCWF evolution returned the wrong times")
    check(length(states) == length(times), "MCWF evolution returned the wrong state count")
    check(isapprox(norm(states[end]), 1; atol=1e-10), "MCWF evolution returned an unnormalized state")
    return real(expect(hamiltonian, states[end]))
end

function steady_state()
    basis = SpinBasis(1 // 2)
    hamiltonian = dense(0.2 * sigmax(basis))
    jump = dense(sigmam(basis))

    state = steadystate.eigenvector(hamiltonian, [jump])
    residual = liouvillian(hamiltonian, [jump]) * state

    check(isapprox(tr(state), 1; atol=1e-10), "steady state is not normalized")
    check(norm(residual) < 1e-8, "steady state does not solve the master equation")
    return real(expect(sigmaz(basis), state))
end

function correlation()
    basis = FockBasis(4)
    annihilation = destroy(basis)
    hamiltonian = number(basis)
    initial_state = dm(fockstate(basis, 0))
    times = [0.0, 0.1, 0.2]

    values = timecorrelations.correlation(
        times,
        initial_state,
        hamiltonian,
        [annihilation],
        annihilation,
        dagger(annihilation),
    )

    check(length(values) == length(times), "correlation returned the wrong value count")
    check(isapprox(values[1], 1; atol=1e-12), "correlation has the wrong initial value")
    check(all(isfinite, values), "correlation returned a non-finite value")
    return real(values[end])
end

const PRECOMPILE_BENCHMARKS = (
    schroedinger=schroedinger,
    master=master,
    mcwf=mcwf,
    steady_state=steady_state,
    correlation=correlation,
)
