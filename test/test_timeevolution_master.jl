using Test
using QuantumOptics
using LinearAlgebra
using OrdinaryDiffEq

@testset "master" begin

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

T = Float64[0.,1.]


fockbasis = FockBasis(10)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint

Ja_unscaled = embed(basis, 1, sm)
Jc_unscaled = embed(basis, 2, destroy(fockbasis))
Junscaled = [Ja_unscaled, Jc_unscaled]

Ja = embed(basis, 1, sqrt(γ)*sm)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jlazy = [LazyTensor(basis, 1, sqrt(γ)*sm), Jc]

Hnh = H - 0.5im*sum([dagger(J[i])*J[i] for i=1:length(J)])
function Ht(t, psi)
    return H*exp(-(5-t)^2), J, dagger.(J)
end
function Hnht(t, psi)
    Hnhtime = H*exp(-(5-t)^2) - 0.5im * sum(i' * i for i in J)
    return Hnhtime, dagger(Hnhtime), J, dagger.(J)
end

Hdense = dense(H)
Hlazy = LazySum(Ha, Hc, Hint)
Hnh_dense = dense(Hnh)
Junscaled_dense = map(dense, Junscaled)
Jdense = map(dense, J)

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
ρ₀ = dm(Ψ₀)

# Test Liouvillian
L = liouvillian(H, J)
ρ = -1im*(H*ρ₀ - ρ₀*H)
for j=J
    ρ .+= j*ρ₀*dagger(j) - 0.5*dagger(j)*j*ρ₀ - 0.5*ρ₀*dagger(j)*j
end
@test tracedistance(L*ρ₀, ρ) < 1e-10

# Test master
@test timeevolution.check_master(ρ₀, Hdense, Jdense, dagger.(Jdense), nothing)
tout, ρt = timeevolution.master(T, ρ₀, Hdense, Jdense; reltol=1e-7)
ρ = ρt[end]
@test tracedistance(exp(dense(L)*T[end])*ρ₀, ρt[end]) < 1e-6

@test timeevolution.check_master(ρ₀, H, J, dagger.(J), nothing)
tout, ρt = timeevolution.master(T, ρ₀, H, J; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, Ψ₀, Hdense, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, Ψ₀, Hlazy, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, Ψ₀, H, Jlazy; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5


# Test master_h
tout, ρt = timeevolution.master_h(T, ρ₀, Hdense, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, H, J; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, H, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, Hdense, J; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, Ψ₀, Hdense, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5


# Test master_nh
tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh, J; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh_dense, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh_dense, J; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, Ψ₀, Hnh_dense, Jdense; reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

# Test no cache derivative methods

t0, t1 = (0.0, 1.0)
fmaster_h!(drho, rho, p, t) = timeevolution.dmaster_h!(drho, H, J, nothing, rho)
prob_master_h! = ODEProblem(fmaster_h!, ρ₀, (t0, t1))
@test_nowarn sol_master_h! = solve(prob_master_h!, DP5(); save_everystep=false)

fmaster_nh!(drho, rho, p, t) = timeevolution.dmaster_nh!(drho, Hnh, J, nothing, rho)
prob_master_nh! = ODEProblem(fmaster_nh!, ρ₀, (t0, t1))
@test_nowarn sol_master_nh! = solve(prob_master_nh!, DP5(); save_everystep=false)

fmaster_h_dynamic!(drho, rho, p, t) = timeevolution.dmaster_h_dynamic!(drho, Ht, nothing, rho, t)
prob_master_h_dynamic! = ODEProblem(fmaster_h_dynamic!, ρ₀, (t0, t1))
@test_nowarn sol_master_h_dynamic! = solve(prob_master_h_dynamic!, DP5(); save_everystep=false)

fmaster_nh_dynamic!(drho, rho, p, t) = timeevolution.dmaster_nh_dynamic!(drho, Hnht, nothing, rho, t)
prob_master_nh_dynamic! = ODEProblem(fmaster_nh_dynamic!, ρ₀, (t0, t1))
@test_nowarn sol_master_nh_dynamic! = solve(prob_master_nh_dynamic!, DP5(); save_everystep=false)

# Test explicit gamma vector
rates_vector = [γ, κ]

tout, ρt = timeevolution.master(T, ρ₀, Hdense, Junscaled_dense; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, ρ₀, H, Junscaled_dense; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, ρ₀, H, Junscaled; rates=rates_vector, reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, H, Junscaled_dense; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh, Junscaled_dense; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, H, Junscaled; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh, Junscaled; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, H, Junscaled; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh, Junscaled; rates=rates_vector, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5


# Test explicit gamma matrix
alpha = 0.3
R = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)]
Rt = transpose(R)
Jrotated_dense = [R[1,1]*Junscaled_dense[1] + R[1,2]*Junscaled_dense[2], R[2,1]*Junscaled_dense[1] + R[2,2]*Junscaled_dense[2]]
Jrotated = [SparseOperator(j) for j=Jrotated_dense]
rates_matrix = diagm(0 => rates_vector)
rates_matrix_rotated = R * rates_matrix * Rt

tout, ρt = timeevolution.master(T, ρ₀, Hdense, Jrotated_dense; rates=rates_matrix_rotated, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master(T, ρ₀, H, Jrotated; rates=rates_matrix_rotated, reltol=1e-6)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_h(T, ρ₀, Hdense, Jrotated_dense; rates=rates_matrix_rotated, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hnh_dense, Jrotated_dense; rates=rates_matrix_rotated, reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5


# Test special cases
tout, ρt = timeevolution.master(T, ρ₀, Hdense, []; reltol=1e-7)
ρ = ρt[end]

tout, ρt = timeevolution.master_h(T, ρ₀, Hdense, []; reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, ρt = timeevolution.master_nh(T, ρ₀, Hdense, []; reltol=1e-7)
@test tracedistance(ρt[end], ρ) < 1e-5

tout, Ψket_t = timeevolution.schroedinger(T, Ψ₀, Hdense; reltol=1.e-7)
tout, Ψbra_t = timeevolution.schroedinger(T, dagger(Ψ₀), Hdense; reltol=1.e-7)
@test tracedistance(Ψket_t[end]⊗Ψbra_t[end], ρ) < 1e-5

L = liouvillian(Hdense, J)
tout, rho_t = timeevolution.master(T, Ψ₀, L)
tout, h_exp = timeevolution.master(T, dm(Ψ₀), L; fout=(t,rho)->expect(Hdense, rho)::ComplexF64)
@test isequal(expect(Hdense, rho_t), h_exp)

end # testset
