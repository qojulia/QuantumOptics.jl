import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_master_timedependent_particle[cython]"

samples = 3
evals = 1
cutoffs = range(5, 41, 5)


def setup(N):
    xmin = -5
    xmax = 5
    x0 = 0.3
    p0 = -0.2
    sigma0 = 1
    dx = (xmax - xmin) / N
    pmin = -np.pi / dx
    pmax = np.pi / dx
    dp = (pmax - pmin) / N

    samplepoints_x = np.linspace(xmin, xmax, N, endpoint=False)
    samplepoints_p = np.linspace(pmin, pmax, N, endpoint=False)

    x = qt.Qobj(np.diag(samplepoints_x))
    row0 = [sum([p*np.exp(-1j*p*dxji) for p in samplepoints_p])*dp*dx/(2*np.pi) for dxji in samplepoints_x - xmin]
    row0 = np.array(row0)
    col0 = row0.conj()

    a = np.zeros([N, N], dtype=complex)
    for i in range(N):
        a[i, i:] = row0[:N - i]
        a[i:, i] = col0[:N - i]
    p = qt.Qobj(a)

    Hkin = p**2
    Vx = 2 * x**2

    def gaussianstate(x0, p0, sigma0):
        alpha = 1./(np.pi**(1/4)*np.sqrt(sigma0))*np.sqrt(dx)
        data = alpha*np.exp(1j*p0*(samplepoints_x-x0/2) - (samplepoints_x-x0)**2/(2*sigma0**2))
        return qt.Qobj(data)

    psi0 = gaussianstate(x0, p0, sigma0)
    J = [x + 1j * p]

    options = qt.Options()
    options.nsteps = 1000000
    options.atol = 1e-8
    options.rtol = 1e-6
    options.rhs_reuse = True

    return psi0, Hkin, Vx, J, x, options


def f(psi0, Hkin, Vx, J, x, options):
    tlist = np.linspace(0, 10, 11)
    H = [Hkin,
            [Vx, "1 + np.cos(t)"]
        ]
    exp_x = qt.mesolve(H, psi0, tlist, J, [x], options=options).expect[0]
    return exp_x


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi0, Hkin, Vx, J, x, options = setup(N)
    checks[N] = sum(f(psi0, Hkin, Vx, J, x, options))
    t = benchmarkutils.run_benchmark(f, psi0, Hkin, Vx, J, x, options, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks, 1e-4)
benchmarkutils.save(name, results)
