import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_mcwf_cavity"

samples = 2
evals = 100
cutoffs = range(50, 501, 50)


def setup(N):
    options = qt.Options()
    options.atol = 1e-8
    options.rtol = 1e-6
    options.num_cpus = 1
    return options


def f(N, options):
    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tspan = np.linspace(0, 10, 11)

    a = qt.destroy(N)
    at = qt.create(N)
    n = at * a

    H = delta_c * n + eta * (a + at)
    J = [np.sqrt(kappa) * a]

    psi0 = qt.coherent(N, alpha0)
    exp_n = qt.mcsolve(H, psi0, tspan, J, [n], ntraj=evals, options=options).expect[0]
    return np.real(exp_n)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    options = setup(N)
    checks[N] = sum(f(N, options))
    t = benchmarkutils.run_benchmark(f, N, options, samples=samples, evals=1)
    results.append({"N": N, "t": t / evals})
print()

benchmarkutils.check(name, checks, eps=0.05)
benchmarkutils.save(name, results)
