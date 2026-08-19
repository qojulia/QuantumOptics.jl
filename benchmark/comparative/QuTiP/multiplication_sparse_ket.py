import scipy.sparse as sp
import qutip as qt
import benchmarkutils

basename = "multiplication_sparse_ket"

samples = 2
evals = 5
cutoffs = range(100, 1001, 100)
S = [0.1, 0.01, 0.001]
Nrand = 5


def setup(N, s):
    data = sp.rand(N, N, s, dtype=float) * 0.2j
    op1 = qt.Qobj(data)
    psi = qt.rand_ket(N).full().ravel()
    return op1, psi


def f(op1, psi):
    return op1 * psi


for s in S:
    name = basename + "_" + str(s).replace(".", "")
    print("Benchmarking:", name)
    print("Cutoff: ", end="", flush=True)
    results = []
    for N in cutoffs:
        print(N, "", end="", flush=True)
        T = 0.
        for i in range(Nrand):
            op1, psi = setup(N, s)
            T += benchmarkutils.run_benchmark(f, op1, psi,
                                              samples=samples, evals=evals)
        results.append({"N": N, "t": T / Nrand})
    print()
    benchmarkutils.save(name, results)
