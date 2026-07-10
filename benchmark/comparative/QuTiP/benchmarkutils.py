import timeit
import json
import qutip
import numpy as np

benchmark_directory = "benchmarks-QuTiP"
commitID = qutip.version.version
result_path = "../results/results-QuTiP-{}-{}.json"

def examplename(name):
    if name.endswith("]"):
        return name.rsplit("[", 1)[0]
    else:
        return name


def run_benchmark(f, *args, samples=5, evals=1):
    D = {"f": f, "args": args}
    t = timeit.repeat("f(*args)", globals=D, number=evals, repeat=samples)
    return min(t)/evals


def check(name, D, eps=1e-5):
    f = open("../checks/" + examplename(name) + ".json")
    data = json.load(f)
    for (N, result) in D.items():
        r = data[str(N)]
        if np.isnan(result) or abs(result-r)/abs(r) > eps:
            print("Warning: Result may be incorrect in", name, ": ", result, "<->", r)


def save(name, results):
    f = open(result_path.format(commitID, name), "w")
    json.dump(results, f)
    f.close()
