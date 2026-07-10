import qutip as qt
import benchmarkutils

name = "multiplication_bra_dense"

samples = 2
evals = 5
cutoffs = range(50, 501, 50)


def setup(N):
    op1 = qt.rand_dm(N, N) * 0.2j
    psi = qt.rand_ket(N).dag().full().ravel()
    return op1, psi


def f(op1, psi):
    return psi * op1


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op1, psi = setup(N)
    t = benchmarkutils.run_benchmark(f, op1, psi,
                                     samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()
benchmarkutils.save(name, results)
