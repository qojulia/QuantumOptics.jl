import qutip as qt
import numpy as np
import benchmarkutils

name = "ptrace_state"

samples = 5
evals = 100
cutoffs = range(10, 51, 10)


def setup(N):
    psi = qt.tensor(qt.coherent(N, 2.0), qt.coherent(N, 1.0))
    return psi


def f(psi):
    return psi.ptrace(0)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi = setup(N)
    rho = f(psi)
    checks[N] = rho.tr()
    t = benchmarkutils.run_benchmark(f, psi, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
