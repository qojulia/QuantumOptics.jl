#!/usr/bin/env bash
while getopts t:v:n:r: flag
do
    case "${flag}" in
        t) tag=${OPTARG};;
        v) jlversion=${OPTARG};;
        n) nthreads=${OPTARG};;
        r) retune=${OPTARG};;
    esac
done
juliaup default $jlversion
mydir=$(mktemp -d)
cp benchmarks.jl $mydir
git checkout $tag --force
rm -f ../Manifest.toml
export JULIA_NUM_THREADS=$nthreads
cat << EOF | julia -t$nthreads
using Pkg;
Pkg.activate(;temp=true);
pkg"add BenchmarkTools OrdinaryDiffEq StochasticDiffEq PkgBenchmark";
pkg"develop .."
using PkgBenchmark
import QuantumOptics
benchmarkpkg(
    QuantumOptics,
    "$tag";
    retune=$retune,
    resultfile="tag=$tag-nthreads=$nthreads-julia=$jlversion.benchmarkresults",
    script="$mydir/benchmarks.jl"
)
EOF
git checkout master --force