using Pkg
# Install the stopwatch tools
Pkg.add("BenchmarkTools")
# Tell Julia to test the local code we just downloaded, not the internet version
Pkg.develop(PackageSpec(path=pwd())) 

using BenchmarkTools
using QuantumOptics

println("Setting up the Quantum Physics Benchmarks...")
const SUITE = BenchmarkGroup()

# --- 1. SCHROEDINGER EVOLUTION ---
SUITE["schroedinger"] = BenchmarkGroup()
basis_spin = SpinBasis(1//2)
H_spin = sigmax(basis_spin)
psi0_spin = spindown(basis_spin)
tspan = [0.0:0.1:1.0;]
SUITE["schroedinger"]["spin_half"] = @benchmarkable timeevolution.schroedinger($tspan, $psi0_spin, $H_spin)

# --- 2. LINDBLAD MASTER EQUATION ---
# This simulates an open quantum system losing energy to its environment
SUITE["lindblad"] = BenchmarkGroup()
decay_rate = 0.1
J = [sqrt(decay_rate) * sigmam(basis_spin)]
rho0 = dm(spinup(basis_spin)) # Initial density matrix (quantum state with classical probability)
SUITE["lindblad"]["decay"] = @benchmarkable timeevolution.master($tspan, $rho0, $H_spin, $J)

# --- 3. MONTE CARLO WAVEFUNCTION (MCWF) ---
# This simulates random quantum jumps (highly computationally expensive)
SUITE["mcwf"] = BenchmarkGroup()
SUITE["mcwf"]["jumps"] = @benchmarkable timeevolution.mcwf($tspan, $psi0_spin, $H_spin, $J; display_beforeevent=false, display_afterevent=false)

println("Tuning and running the benchmarks (this will take a few minutes)...")
tune!(SUITE)
results = run(SUITE, verbose=true)

println("\n========== FINAL BENCHMARK RESULTS ==========")
for (k, v) in results
    println("--- ", k, " ---")
    display(v)
end
