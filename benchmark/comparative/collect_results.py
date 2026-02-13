#!/usr/bin/env python3
"""Collect benchmark results from individual JSON files into consolidated files.

For each benchmark name, produces a single JSON file in results-collected/
that maps framework-version to the list of {N, t} data points.
"""

import os
import json

names = [
    "coherentstate",
    "expect_operator",
    "ptrace_state",
    "qfunc_state",
    "wigner_state",
    "timeevolution_schroedinger_cavity",
    "timeevolution_master_jaynescummings",
    "timeevolution_mcwf_cavity",
]

filenames = os.listdir("results")


def extract_version(filename, testname):
    name, _ = os.path.splitext(filename)
    if name.endswith("]"):
        name, variant = name.rsplit("[", 1)
        variant = "/" + variant[:-1]
    else:
        variant = ""
    assert name.startswith("results-"), name
    assert name.endswith("-" + testname), name
    return name[len("results-"):-len("-" + testname)] + variant


def cutdigits(x):
    return float("%.3g" % (x))


for testname in names:
    print("Loading: ", testname)
    matches = [f for f in filenames if testname in f and f.endswith(".json") and f != "specs.json"]
    d = {}
    for filename in matches:
        version = extract_version(filename, testname)
        with open("results/" + filename) as f:
            data = json.load(f)
        for point in data:
            point["t"] = cutdigits(point["t"])
        d[version] = data
    path = "results-collected/" + testname + ".json"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

print("\nCollected results written to results-collected/")
