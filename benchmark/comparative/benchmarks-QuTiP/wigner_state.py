import qutip as qt
import numpy as np
import benchmarkutils

name = "wigner_state"

samples = 3
evals = 10
cutoffs = range(50, 501, 50)


def setup(N):
    psi = qt.coherent(N, 3.0)
    xvec = np.linspace(-5, 5, 50)
    yvec = np.linspace(-5, 5, 50)
    return psi, xvec, yvec


def f(psi, xvec, yvec):
    return qt.wigner(psi, xvec, yvec)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi, xvec, yvec = setup(N)
    checks[N] = np.sum(f(psi, xvec, yvec))
    t = benchmarkutils.run_benchmark(f, psi, xvec, yvec, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
