import qutip as qt
import numpy as np
import benchmarkutils

name = "timeevolution_master_particle"

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

    H = p**2 + 2 * x**2

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

    return psi0, H, x, J, options


def f(psi0, H, x, J, options):
    tlist = np.linspace(0, 10, 11)
    exp_x = qt.mesolve(H, psi0, tlist, J, [x], options=options).expect[0]
    return exp_x


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi0, H, x, J, options = setup(N)
    checks[N] = sum(f(psi0, H, x, J, options))
    t = benchmarkutils.run_benchmark(f, psi0, H, x, J, options, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks, 1e-4)
benchmarkutils.save(name, results)
