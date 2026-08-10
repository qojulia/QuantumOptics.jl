using Pkg

docs_dir = @__DIR__
package_dir = normpath(joinpath(docs_dir, ".."))
examples_dir = joinpath(docs_dir, "examples")

function build_examples()
    Pkg.activate(examples_dir)
    Pkg.develop(PackageSpec(path = package_dir))
    Pkg.instantiate()

    include(joinpath(examples_dir, "make.jl"))
    Pkg.activate(docs_dir)
end

build_examples()

using Documenter
using AnythingLLMDocs
using QuantumInterface
using QuantumOptics
using QuantumOpticsBase

pages = [
    "index.md",
    "installation.md",
    "tutorial.md",
    "Quantum objects" => [
        "Introduction" => "quantumobjects/quantumobjects.md",
        "quantumobjects/bases.md",
        "quantumobjects/states.md",
        "quantumobjects/operators.md",
    ],
    "Quantum systems" => [
        "Introduction" => "quantumsystems/quantumsystems.md",
        "quantumsystems/spin.md",
        "quantumsystems/fock.md",
        "quantumsystems/charge.md",
        "quantumsystems/nlevel.md",
        "quantumsystems/particle.md",
        "quantumsystems/subspace.md",
        "quantumsystems/manybody.md",
    ],
    "Time-evolution" => [
        "Introduction" => "timeevolution/timeevolution.md",
        "Schroedinger equation" => "timeevolution/schroedinger.md",
        "Master equation" => "timeevolution/master.md",
        "Quantum trajectories" => "timeevolution/mcwf.md",
        "Time-dependent problems" => "timeevolution/timedependent-problems.md",
    ],
    "metrics.md",
    "steadystate.md",
    "timecorrelations.md",
    "semiclassical.md",
    "Stochastics" => [
        "Introduction" => "stochastic/stochastic.md",
        "Stochastic Schroedinger equation" => "stochastic/schroedinger.md",
        "Stochastic Master equation" => "stochastic/master.md",
        "Stochastic semiclassical systems" => "stochastic/semiclassical.md",
    ],
    "Examples" => [
        "Pumped cavity" => "examples/pumped-cavity.md",
        "Jaynes-Cummings" => "examples/jaynes-cummings.md",
        "Superradiant laser" => "examples/superradiant-laser.md",
        "Particle in harmonic trap" => "examples/particle-in-harmonic-trap.md",
        "Particle into barrier" => "examples/particle-into-barrier.md",
        "Wavepacket in 2D" => "examples/wavepacket2D.md",
        "Raman transition" => "examples/raman.md",
        "2 qubit entanglement" => "examples/two-qubit-entanglement.md",
        "Correlation spectrum" => "examples/correlation-spectrum.md",
        "Simple many-body system" => "examples/manybody-fourlevel-system.md",
        "N particles in double well" => "examples/nparticles-in-double-well.md",
        "Doppler cooling" => "examples/doppler-cooling.md",
        "Cavity cooling" => "examples/cavity-cooling.md",
        "Dipole-coupled nanorings" => "examples/atomic_dipole_arrays.md",
        "Lasing and cooling" => "examples/lasing-and-cooling.md",
        "Heat-pumped Maser" => "examples/three-level-maser.md",
        "Optomechanical cavity" => "examples/optomech-cooling.md",
        "Ramsey spectroscopy" => "examples/ramsey.md",
        "Dephasing of Atom" => "examples/atom-dephasing.md",
        "Quantum Zeno Effect" => "examples/quantum-zeno-effect.md",
        "Quantum Kicked Top" => "examples/quantum-kicked-top.md",
        "Quantum Vortices" => "examples/vortex.md",
        "Spinor BEC" => "examples/spin-orbit-coupled-BEC1D.md",
    ],
    "api.md",
]

doc_modules = [QuantumInterface, QuantumOptics, QuantumOpticsBase]

api_base = "https://anythingllm.krastanov.org/api/v1"
anythingllm_assets = integrate_anythingllm(
    "QuantumOptics",
    doc_modules,
    docs_dir,
    api_base;
    repo = "github.com/qojulia/QuantumOptics.jl.git",
    options = EmbedOptions(),
)

makedocs(
    sitename = "QuantumOptics.jl",
    modules = doc_modules,
    pages = pages,
    format = Documenter.HTML(
        assets = anythingllm_assets,
        canonical = "https://docs.qojulia.org/",
        size_threshold = 400 * 2^10,
        size_threshold_warn = 300 * 2^10,
    ),
    checkdocs = :public,
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/qojulia/QuantumOptics.jl.git",
    devbranch = "master",
    deploy_config = Documenter.Buildkite(),
    push_preview = true,
)
