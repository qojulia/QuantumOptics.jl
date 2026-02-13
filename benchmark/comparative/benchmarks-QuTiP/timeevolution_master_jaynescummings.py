import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_master_jaynescummings"

samples = 3
evals = 3
cutoffs = range(10, 51, 10)


def setup(N):
    options = {"atol": 1e-8, "rtol": 1e-6}
    return options


def f(N, options):
    wc = 1.0
    wa = 1.0
    g = 0.5
    kappa = 0.1
    gamma = 0.05
    tspan = np.linspace(0, 10, 11)

    a = qt.tensor(qt.destroy(N), qt.qeye(2))
    sm = qt.tensor(qt.qeye(N), qt.sigmam())
    sp = qt.tensor(qt.qeye(N), qt.sigmap())
    n = a.dag() * a

    H = wc * a.dag() * a + wa * sp * sm + g * (a.dag() * sm + a * sp)
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sm]

    psi0 = qt.tensor(qt.fock(N, 1), qt.basis(2, 0))
    exp_n = qt.mesolve(H, psi0, tspan, c_ops, [n], options=options).expect[0]
    return np.real(exp_n)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    options = setup(N)
    checks[N] = sum(f(N, options))
    t = benchmarkutils.run_benchmark(f, N, options, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
