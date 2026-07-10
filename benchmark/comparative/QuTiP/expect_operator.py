import qutip as qt
import numpy as np
import benchmarkutils

name = "expect_operator"

samples = 5
evals = 100
cutoffs = range(100, 2501, 100)

def setup(N):
    op = (qt.destroy(N) + qt.create(N))
    psi = qt.Qobj(np.ones(N, complex)/(N**(1/2)))
    rho = psi*psi.dag()
    return op, rho

def f(op, rho):
    return qt.expect(op, rho)

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op, rho = setup(N)
    checks[N] = f(op, rho)
    t = benchmarkutils.run_benchmark(f, op, rho, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
