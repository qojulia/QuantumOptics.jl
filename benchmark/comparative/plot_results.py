import os
import json
import pylab

sourcedir = "results-collected"


def transform_version(points):
    Nvec = []
    tvec = []
    for x in points:
        Nvec.append(x["N"])
        tvec.append(x["t"])
    return Nvec, tvec


def transform_json(d):
    data = []
    for name, points in d.items():
        data.append((name, transform_version(points)))
    data.sort(key=lambda x: x[0])
    return data


testnames = os.listdir(sourcedir)
for testname in testnames:
    if testname.startswith("."):
        continue
    # if not "ptrace" in testname:
    #     continue
    print("Open ", testname)
    f = open(os.path.join(sourcedir, testname))
    d = json.load(f)
    f.close()
    data = transform_json(d)
    pylab.figure()
    for name, points in data:
        pylab.title(testname)
        pylab.plot(points[0], points[1], label=name)
        pylab.plot(points[0], points[1], "ok", alpha=0.4)
    pylab.legend()
pylab.show()


