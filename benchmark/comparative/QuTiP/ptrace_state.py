import qutip as qt
import numpy as np
import benchmarkutils

name = "ptrace_state"

samples = 5
evals = 100
cutoffs = range(2, 16)


def setup(N):
    def create_substate(c0, alpha, N):
        x = np.linspace(0., 1., N)
        data = (c0 + alpha * x).conj()
        return qt.Qobj(data)
    psi1 = create_substate(1, 0.2, N)
    psi2 = create_substate(-2, 0.3, N)
    psi3 = create_substate(3, 0.4, 2)
    psi4 = create_substate(4, 0.5, 2)
    psi = qt.tensor(psi1, psi2, psi3, psi4)
    return psi


def f(psi):
    return qt.ptrace(psi, [0, 3])


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi = setup(N)
    checks[N] = abs(f(psi).full()).sum()
    t = benchmarkutils.run_benchmark(f, psi, samples=samples, evals=evals)
    results.append({"N": 4 * N**2, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
