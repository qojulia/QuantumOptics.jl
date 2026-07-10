import scipy.sparse as sp
import benchmarkutils

basename = "multiplication_sparse_sparse"

samples = 2
evals = 5
cutoffs = range(50, 1001, 50)
S = [0.1, 0.01, 0.001]
Nrand = 5


def setup(N, s):
    op1 = sp.rand(N, N, s, dtype=float) * 0.2j
    op2 = sp.rand(N, N, s, dtype=float) * 0.1j
    return op1, op2


def f(op1, op2):
    return op1 * op2


for s in S:
    name = basename + "_" + str(s).replace(".", "")
    print("Benchmarking:", name)
    print("Cutoff: ", end="", flush=True)
    results = []
    for N in cutoffs:
        print(N, "", end="", flush=True)
        T = 0.
        for i in range(Nrand):
            op1, op2 = setup(N, s)
            T += benchmarkutils.run_benchmark(f, op1, op2,
                                              samples=samples, evals=evals)
        results.append({"N": N, "t": T / Nrand})
    print()
    benchmarkutils.save(name, results)
