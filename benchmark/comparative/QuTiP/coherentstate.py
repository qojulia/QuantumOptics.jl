import qutip as qt
import numpy as np
import benchmarkutils

name = "coherentstate"

samples = 5
evals = 100
cutoffs = range(50, 501, 50)

def setup(N):
    alpha = np.log(N)
    return alpha

def f(N, alpha):
    return qt.coherent(N, alpha, method="analytic")

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    alpha = setup(N)
    checks[N] = qt.expect(qt.destroy(N), f(N, alpha))
    t = benchmarkutils.run_benchmark(f, N, alpha, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks, 0.05)
benchmarkutils.save(name, results)
