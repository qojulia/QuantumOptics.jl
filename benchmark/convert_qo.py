import os

d = "comparative/QuantumOptics.jl"
for f in os.listdir(d):
    if not f.endswith(".jl") or f == "benchmarkutils.jl":
        continue
    
    path = os.path.join(d, f)
    with open(path, "r") as file:
        content = file.read()
    
    # We want to replace the benchmarking loop.
    # Usually it's:
    # println("Benchmarking: ", name)
    # print("Cutoff: ")
    # results = []
    # for N in cutoffs
    # ...
    # end
    # println()
    # benchmarkutils.save(name, results)
    
    if "isdefined(Main, :RUN_CI)" in content:
        continue # already converted
        
    parts = content.split('println("Benchmarking: ", name)')
    if len(parts) == 2:
        new_loop = """
if isdefined(Main, :RUN_CI)
    SUITE[name] = BenchmarkGroup()
    for N in (length(cutoffs) > 0 ? [cutoffs[1]] : [])
        setup_args = setup(N)
        # some setups return a tuple, some return a single value
        if setup_args isa Tuple
            SUITE[name][string(N)] = @benchmarkable f($(setup_args)...)
        else
            SUITE[name][string(N)] = @benchmarkable f($setup_args)
        end
    end
else
    println("Benchmarking: ", name)
""" + parts[1] + "\nend\n"
        with open(path, "w") as file:
            file.write(parts[0] + new_loop)
