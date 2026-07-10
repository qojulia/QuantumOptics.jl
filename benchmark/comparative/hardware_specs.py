import subprocess
import json
import sys

output_path = "results/specs.json"

specs = {}

print("Running lspcu ...")
data = subprocess.check_output(["lscpu"])
cpuinfo = data.decode(sys.getdefaultencoding())
cpuspecs = []
specs["cpu"] = cpuspecs

for line in cpuinfo.split("\n"):
    if line:
        key, value = line.split(":")
        cpuspecs.append((key, value.strip()))


print("Getting julia specs ...")
data = subprocess.check_output(["julia", "-e", "import InteractiveUtils; InteractiveUtils.versioninfo()"])
juliainfo = data.decode(sys.getdefaultencoding())
juliaspecs = []
specs["julia"] = juliaspecs

for line in juliainfo.split("\n"):
    if line:
        juliaspecs.append(line.strip())

print("Getting qutip specs ...")
data = subprocess.check_output(["python3", "-c", "import qutip; qutip.about()"])
qutipinfo = data.decode(sys.getdefaultencoding())
qutipspecs = []
specs["qutip"] = qutipspecs

for line in qutipinfo.split("\n")[4:]:
    if line:
        qutipspecs.append(line.strip())


f = open(output_path, "w")
f.write(json.dumps(specs))
f.close()
