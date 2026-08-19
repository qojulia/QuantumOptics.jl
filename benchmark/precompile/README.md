The workloads in `scenarios.jl` are run by [`QuantumSavory/julia-precompile-benchmark@v3`](https://github.com/QuantumSavory/julia-precompile-benchmark); the action owns dependency isolation, measurement, and summarization. For a local comparison, download the matching v3.0.0 tool bundle and run it from this repository:

```sh
gh release download v3.0.0 --repo QuantumSavory/julia-precompile-benchmark --pattern julia-precompile-benchmark-v3.0.0.tar.gz
tar -xzf julia-precompile-benchmark-v3.0.0.tar.gz
julia-precompile-benchmark-v3.0.0/run.sh benchmark/precompile/scenarios.jl /tmp/precompile-results base=/path/to/base head=/path/to/head
```
