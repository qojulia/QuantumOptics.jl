import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_mcwf_jaynescummings"

samples = 2
evals = 100
cutoffs = range(50, 501, 50)


def setup(N):
    options = qt.Options()
    options.atol = 1e-8
    options.rtol = 1e-6
    options.num_cpus = 1
    return options


def f(N, options):
    wa = 1
    wc = 1
    g = 2
    kappa = 0.5
    gamma = 0.1
    n_th = 0.75
    tspan = np.linspace(0, 10, 11)

    Ia = qt.qeye(2)
    Ic = qt.qeye(N)

    a = qt.destroy(N)
    at = qt.create(N)
    n = at * a

    sm = qt.sigmam()
    sp = qt.sigmap()
    sz = qt.sigmaz()

    H = wc*qt.tensor(n, Ia) + qt.tensor(Ic, wa/2.*sz) + g*(qt.tensor(at, sm) + qt.tensor(a, sp))
    c_ops = [
        qt.tensor(np.sqrt(kappa*(1+n_th)) * a, Ia),
        qt.tensor(np.sqrt(kappa*n_th) * at, Ia),
        qt.tensor(Ic, np.sqrt(gamma) * sm),
    ]

    psi0 = qt.tensor(qt.fock(N, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    exp_n = qt.mcsolve(H, psi0, tspan, c_ops, [qt.tensor(n, Ia)], ntraj=evals, options=options).expect[0]
    return np.real(exp_n)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    options = setup(N)
    checks[N] = sum(f(N, options))
    t = benchmarkutils.run_benchmark(f, N, options, samples=samples, evals=1)
    results.append({"N": 2*N, "t": t / evals})
print()

benchmarkutils.check(name, checks, eps=0.05)
benchmarkutils.save(name, results)
