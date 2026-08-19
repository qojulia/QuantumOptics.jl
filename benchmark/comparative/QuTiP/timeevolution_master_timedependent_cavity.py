import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_master_timedependent_cavity"

samples = 3
evals = 1
cutoffs = range(10, 131, 10)


def setup(N):
    options = qt.Options()
    options.nsteps = 1000000
    options.atol = 1e-8
    options.rtol = 1e-6
    return options


def f(N, options):
    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    # delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tspan = np.linspace(0, 10, 11)

    a = qt.destroy(N)
    at = qt.create(N)
    n = qt.num(N)

    J = [np.sqrt(kappa) * a]
    psi0 = qt.coherent(N, alpha0)
    H = [wc * n,
            [eta * a, lambda t, args: np.exp(1j * wl * t)],
            [eta * at, lambda t, args: np.exp(-1j * wl * t)]]
    alpha_t = qt.mesolve(H, psi0, tspan, J, [a], options=options).expect[0]
    return np.real(alpha_t)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    options = setup(N)
    checks[N] = sum(f(N, options))
    t = benchmarkutils.run_benchmark(f, N, options, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks, 1e-4)
benchmarkutils.save(name, results)
