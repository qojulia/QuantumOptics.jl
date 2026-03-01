import timeit
import json
import qutip
import numpy as np

commitID = qutip.__version__
result_path = "../results/results-QuTiP-{}-{}.json"


def examplename(name):
    if name.endswith("]"):
        return name.rsplit("[", 1)[0]
    else:
        return name


def run_benchmark(f, *args, samples=5, evals=1):
    D = {"f": f, "args": args}
    t = timeit.repeat("f(*args)", globals=D, number=evals, repeat=samples)
    return min(t) / evals


def check(name, D, eps=1e-5):
    check_path = "../checks/" + examplename(name) + ".json"
    try:
        with open(check_path) as f:
            data = json.load(f)
        for N, result in D.items():
            r = data[str(N)]
            if np.isnan(result) or abs(result - r) / abs(r) > eps:
                print(f"Warning: Result may be incorrect in {name}: {result} <-> {r}")
    except FileNotFoundError:
        print("No check file found - write results to check file.")
        with open(check_path, "w") as f:
            json.dump(D, f)


def save(name, results):
    path = result_path.format(commitID, name)
    with open(path, "w") as f:
        json.dump(results, f)
