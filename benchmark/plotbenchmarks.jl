using CairoMakie
using AlgebraOfGraphics
using QuantumClifford
using PkgBenchmark
using DataFrames
using DataFramesMeta
using Statistics
using Glob

function results_to_dataframe(res)
    recs = []
    bmg = res.benchmarkgroup
    for taxon1 in keys(bmg)
        for taxon2 in keys(bmg[taxon1])
            for taxon3 in keys(bmg[taxon1][taxon2])
                best = minimum(bmg[taxon1][taxon2][taxon3])
                time = best.time
                memory = best.memory
                allocs = best.allocs
                push!(recs, (;taxon1,taxon2,taxon3,time,memory,allocs))
            end
        end
    end
    DataFrame(recs)
end

##
cd("QuantumCliffordBenchmarksLog/logs")
##

files = glob("*.benchmarkresults")
subframes = []
for f in files
    m = match(r"tag=(v[\d\.]*)-nthreads=(\d)-julia=([\d\.\w]*).benchmarkresults", f)
    if isnothing(m)
        @warn "problem with $(f)"
        continue
    end
    sdf = results_to_dataframe(readresults(f))
    #sdf[!,:tag] .= m.captures[1]
    sdf[!,:version] .= VersionNumber(m.captures[1][2:end])
    sdf[!,:threads] .= m.captures[2]
    #sdf[!,:julia] .= m.captures[3]
    sdf[!,:julia] .= m.captures[3]=="nightly" ? v"1.999" : VersionNumber(m.captures[3])
    push!(subframes, sdf)
end

##

df = vcat(subframes...)
maxjversion = maximum(df[df.julia .!= v"1.999",:julia])
df[df.julia .== v"1.999", :julia] .= VersionNumber(maxjversion.major, maxjversion.minor+1, 0)
df[!,:fullname] .= df[:,:taxon1] .* "-" .* df[:,:taxon2] .* "-" .* df[:,:taxon3]
df[!,:groupname] .= df[:,:taxon1] .* "-" .* df[:,:taxon2]
df[!,:logtime] .= log10.(df[:,:time])
df_min = combine(groupby(df, [:taxon3, :taxon2, :taxon1, :julia, :threads]), :logtime=>minimum=>:minlogtime)
df = outerjoin(df, df_min, on=[:taxon3, :taxon2, :taxon1, :julia, :threads])
df[!,:normlogtime] .= df[:,:logtime] .- df[:,:minlogtime]

df2 = combine(groupby(df, [:groupname, :taxon1, :version, :julia, :threads]), :normlogtime=>mean=>:logtime)
df1 = combine(groupby(df2, [:taxon1, :version, :julia, :threads]), :logtime=>mean=>:logtime)
df0 = combine(groupby(df1, [:version, :julia, :threads]), :logtime=>mean=>:logtime)

layers = visual(Stairs, step=:center)
axis = (;width = 500, height = 400, xticklabelrotation=-pi/3)
facet = (;linkyaxes=:minimal)

##

benchmark_data3 = data(df)
mappings3 = mapping(:version, :time, color=:julia, row=:fullname, col=:threads)
f3 = draw(benchmark_data3 * layers * mappings3; facet, axis)
save("../benchmarks3.png",f3)

benchmark_data2 = data(df2)
mappings2 = mapping(:version, :logtime, color=:julia, row=:groupname, col=:threads)
f2 = draw(benchmark_data2 * layers * mappings2; facet, axis)
save("../benchmarks2.png",f2)

benchmark_data1 = data(df1)
mappings1 = mapping(:version, :logtime, color=:julia, row=:taxon1, col=:threads)
f1 = draw(benchmark_data1 * layers * mappings1; facet, axis)
save("../benchmarks1.png",f1)

benchmark_data0 = data(df0)
mappings0 = mapping(:version, :logtime, color=:julia, col=:threads)
f0 = draw(benchmark_data0 * layers * mappings0; facet, axis)
save("../benchmarks0.png",f0)

mappings3a = mapping(:version, :allocs, color=:julia, row=:fullname, col=:threads)
f3a = draw(benchmark_data3 * layers * mappings3a; facet, axis)
save("../benchmarks3_allocs.png",f3a)
