import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_schroedinger_cavity"

samples = 3
evals = 3
cutoffs = range(50, 501, 50)


def setup(N):
    options = qt.Options()
    options.atol = 1e-8
    options.rtol = 1e-6
    return options


def f(N, options):
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tspan = np.linspace(0, 10, 11)

    a = qt.destroy(N)
    at = qt.create(N)
    n = at * a

    H = delta_c*n + eta*(a + at)

    psi0 = qt.coherent(N, alpha0)
    exp_n = qt.mesolve(H, psi0, tspan, [], [n], options=options).expect[0]
    return np.real(exp_n)


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

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
