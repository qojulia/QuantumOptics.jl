import qutip as qt
import numpy as np
import benchmarkutils

name = "wigner_operator"

samples = 3
evals = 5
cutoffs = range(10, 101, 10)

def setup(N):
    alpha = 0.7
    xvec = np.linspace(-50, 50, 100)
    yvec = np.linspace(-50, 50, 100)
    state = qt.coherent(N, alpha)
    op = state*state.dag()
    return op, xvec, yvec

def f(state, xvec, yvec):
    return qt.wigner(state, xvec, yvec)

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op, xvec, yvec = setup(N)
    alpha_check = 0.6 + 0.1j
    checks[N] = f(op, [alpha_check.real], [alpha_check.imag])[0, 0]
    t = benchmarkutils.run_benchmark(f, op, xvec, yvec, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
