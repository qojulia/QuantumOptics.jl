#!/usr/bin/env python3
"""Collect hardware and software specifications for benchmark reproducibility."""

import subprocess
import json
import sys

output_path = "results/specs.json"

specs = {}

print("Running lscpu ...")
try:
    data = subprocess.check_output(["lscpu"])
    cpuinfo = data.decode(sys.getdefaultencoding())
    cpuspecs = []
    specs["cpu"] = cpuspecs
    for line in cpuinfo.split("\n"):
        if line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                cpuspecs.append((parts[0].strip(), parts[1].strip()))
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Warning: lscpu not available")

print("Getting Julia specs ...")
try:
    data = subprocess.check_output(
        ["julia", "-e", "import InteractiveUtils; InteractiveUtils.versioninfo()"]
    )
    juliainfo = data.decode(sys.getdefaultencoding())
    juliaspecs = []
    specs["julia"] = juliaspecs
    for line in juliainfo.split("\n"):
        if line:
            juliaspecs.append(line.strip())
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Warning: julia not available")

print("Getting QuTiP specs ...")
try:
    data = subprocess.check_output(["python3", "-c", "import qutip; qutip.about()"])
    qutipinfo = data.decode(sys.getdefaultencoding())
    qutipspecs = []
    specs["qutip"] = qutipspecs
    for line in qutipinfo.split("\n")[4:]:
        if line:
            qutipspecs.append(line.strip())
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Warning: qutip not available")

with open(output_path, "w") as f:
    json.dump(specs, f, indent=2)

print(f"Specs written to {output_path}")
