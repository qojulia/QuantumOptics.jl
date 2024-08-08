using Test
using QuantumOptics
using OrdinaryDiffEq

@testset "sciml interface" begin

# semiclassical ODE problem
b = SpinBasis(1//2)
psi0 = spindown(b)
u0 = ComplexF64[0.5, 0.75] 
sc = semiclassical.State(psi0, u0)
t₀, t₁ = (0.0, pi)
σx = sigmax(b)

fquantum(t, q, u) = σx + cos(u[1])*identityoperator(σx)
fclassical!(du, u, q, t) = (du[1] = sin(u[2]); du[2] = 2*u[1])
f!(dstate, state, p, t) = semiclassical.dschroedinger_dynamic!(dstate, fquantum, fclassical!, state, t)
prob = ODEProblem(f!, sc, (t₀, t₁))

sol = solve(prob, DP5(); reltol = 1.0e-8, abstol = 1.0e-10, save_everystep=false)
tout, ψt = semiclassical.schroedinger_dynamic([t₀, t₁], sc, fquantum, fclassical!; reltol = 1.0e-8, abstol = 1.0e-10)

@test sol[end] ≈ ψt[end]

end