module benchmarkutils

using QuantumOptics
using JSON

set_zero_subnormals(true)

# Detect the git commit hash
const rootpath = abspath(".")
const quantumoptics_directory = dirname(dirname(Base.functionloc(QuantumOptics.eval, Tuple{Nothing})[1]))
cd(quantumoptics_directory)
const commitID = read(`git rev-parse --verify HEAD`, String)[1:10]
cd(rootpath)
println("Detected QuantumOptics.jl version: ", commitID)

benchmark_directory = "benchmarks-QuantumOptics.jl"
checkvalues_directory = "checks"

function examplename(name)
    if endswith(name, "]")
        return rsplit(name, "[", limit=2)[1]
    end
    name
end

function check(name, D, eps=1e-5)
    check_path = "../checks/$(examplename(name)).json"
    if ispath(check_path)
        println("Checking against check file.")
        data = JSON.parsefile(check_path)
        for (N, result) in D
            r = data[string(N)]
            if isnan(result) || abs(result-r)/abs(r) > eps
                println("Warning: Result may be incorrect in", name, ": ", result, "<->", r)
            end
        end
    else
        println("No check file found - write results to check file.")
        f = open(check_path, "w")
        write(f, JSON.json(D))
        close(f)
    end
end

function save(name, results)
    result_path = "../results/results-QuantumOptics.jl-$commitID-$name.json"
    f = open(result_path, "w")
    write(f, JSON.json(results))
    close(f)
end

end # module
