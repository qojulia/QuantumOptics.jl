website = "../QuantumOptics.jl-website"
@assert isdir(website)

run(`python3 collect_results.py`)
run(`python3 extract_code.py`)

sourcecode_in = "sourcecode"
sourcecode_out = joinpath(website, "src/_benchmarks-sourcecode/")
data_in = "results-collected"
data_out = joinpath(website, "src/benchmark-data/")

cp(sourcecode_in, sourcecode_out; remove_destination=true)
cp(data_in, data_out; remove_destination=true)
