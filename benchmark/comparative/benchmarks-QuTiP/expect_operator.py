import qutip as qt
import numpy as np
import benchmarkutils

name = "expect_operator"

samples = 5
evals = 1000
cutoffs = range(50, 501, 50)


def setup(N):
    op = qt.num(N)
    psi = qt.coherent(N, 3.0)
    return op, psi


def f(op, psi):
    return qt.expect(op, psi)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op, psi = setup(N)
    checks[N] = f(op, psi)
    t = benchmarkutils.run_benchmark(f, op, psi, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
