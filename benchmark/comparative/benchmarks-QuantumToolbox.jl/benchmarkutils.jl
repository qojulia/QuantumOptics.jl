module benchmarkutils

using QuantumToolbox
using JSON

set_zero_subnormals(true)

const rootpath = abspath(".")
const commitID = try
    string(pkgversion(QuantumToolbox))
catch
    "unknown"
end
println("Detected QuantumToolbox.jl version: ", commitID)

benchmark_directory = "benchmarks-QuantumToolbox.jl"
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
                println("Warning: Result may be incorrect in ", name, ": ", result, " <-> ", r)
            end
        end
    else
        println("No check file found.")
    end
end

function save(name, results)
    result_path = "../results/results-QuantumToolbox.jl-$commitID-$name.json"
    f = open(result_path, "w")
    write(f, JSON.json(results))
    close(f)
end

end # module
