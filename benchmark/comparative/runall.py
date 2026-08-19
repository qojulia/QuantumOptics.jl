import os
import subprocess

juliabenchmarks = "QuantumOptics.jl"
pythonbenchmarks = "QuTiP"
qtoolboxbenchmarks = "QuantumToolbox.jl"

subprocess.run(["python3", "hardware_specs.py"], check=True)

for folder, cmd, ext in [(juliabenchmarks, "julia", ".jl"), (pythonbenchmarks, "python3", ".py"), (qtoolboxbenchmarks, "julia", ".jl")]:
    os.chdir(folder)
    filenames = os.listdir(".")
    for name in filenames:
        if "benchmarkutils" in name or not name.endswith(ext):
            continue
        subprocess.run([cmd, name], check=True)
    os.chdir("..")
