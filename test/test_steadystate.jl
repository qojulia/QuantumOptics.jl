using Test
using QuantumOptics
using LinearAlgebra

@testset "steadystate" begin

ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1
η = 1.5

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
Hdense = dense(H)

Ja = embed(basis, 1, sqrt(γ)*sm)
Ja2 = embed(basis, 1, sqrt(0.5*γ)*sp)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jdense = map(dense, J)

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 2)
ρ₀ = dm(Ψ₀)
ψ0_p = fockstate(fockbasis, 0)
ρ0_p = dm(ψ0_p)

tout, ρt = timeevolution.master([0,100], ρ₀, Hdense, Jdense; reltol=1e-7)

tss, ρss = steadystate.master(Hdense, Jdense; tol=1e-4)
@test tracedistance(ρss[end], ρt[end]) < 1e-3

ρss = steadystate.eigenvector(Hdense, Jdense)
@test tracedistance(ρss, ρt[end]) < 1e-6

ρss = steadystate.eigenvector(Hdense, Jdense, nev = 1)
@test tracedistance(ρss, ρt[end]) < 1e-6

ρss = steadystate.eigenvector(H, sqrt(2).*J; rates=0.5.*ones(length(J)), tol=1e-8)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.eigenvector(H, sqrt(2).*J; rates=0.5.*ones(length(J)), nev = 1)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.eigenvector(liouvillian(Hdense, Jdense))
@test tracedistance(ρss, ρt[end]) < 1e-6

@test_throws TypeError steadystate.eigenvector(H, J; ncv="a")

ev, ops = steadystate.liouvillianspectrum(Hdense, Jdense)
@test tracedistance(ρss, ops[1]) < 1e-12
@test ev[sortperm(abs.(ev))] == ev

ev, ops = steadystate.liouvillianspectrum(H, sqrt(2).*J; rates=0.5.*ones(length(J)), nev = 1)
@test tracedistance(ρss, ops[1]/tr(ops[1])) < 1e-12
@test ev[sortperm(abs.(ev))] == ev

# Test iterative solvers
ρss, h = steadystate.iterative(Hdense, Jdense; log=true)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = copy(ρ₀)
steadystate.iterative!(ρss, H, J)
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.iterative(H, sqrt(2) .* J; rates=0.5.*ones(length(J)))
@test tracedistance(ρss, ρt[end]) < 1e-3

ρss = steadystate.iterative(H, sqrt(2) .* J; rates=diagm(0=>[0.5,0.5]))
@test tracedistance(ρss, ρt[end]) < 1e-3

# Test different float types
ρss32 = Operator(basis, basis, Matrix{ComplexF32}(ρ₀.data))
Hdense32 = Operator(basis, basis, Matrix{ComplexF32}(H.data))
Jdense32 = [Operator(basis, basis, Matrix{ComplexF32}(j.data)) for j ∈ J]
steadystate.iterative!(ρss32, Hdense32, Jdense32)
ρt32 = DenseOperator(ρt[end].basis_l, ρt[end].basis_r, Matrix{ComplexF32}(ρt[end].data))
@test tracedistance(ρss32, ρt[end]) < 2*1e-3

# Iterative methods with lazy operators
Hlazy = LazySum(Ha, Hc, Hint)
Ja_lazy = LazyTensor(basis, 1, sqrt(γ)*sm)
Jc_lazy = LazyTensor(basis, 2, sqrt(κ)*destroy(fockbasis))
Jlazy = [Ja_lazy, Jc_lazy]

ρss = steadystate.iterative(Hlazy, Jlazy)
@test tracedistance(ρss, ρt[end]) < 1e-3

# Make sure initial state never got mutated
@test ρ₀ == dm(Ψ₀)

# Compute steady-state photon number of a driven cavity (analytically: η^2/κ^2)
Hp = η*(destroy(fockbasis) + create(fockbasis))
Jp = [sqrt(2κ)*destroy(fockbasis)]
n_an = η^2/κ^2

tss,ρss = steadystate.master(Hp, Jp; rho0=ρ0_p, tol=1e-4)
nss = expect(create(fockbasis)*destroy(fockbasis), ρss[end])
@test n_an - real(nss) < 1e-3

ρss = steadystate.eigenvector(Hp, Jp)
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3

ρss = steadystate.eigenvector(dense(Hp), map(dense, Jp))
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3

# Test iterative solvers
ρss = steadystate.iterative(dense(Hp), dense.(Jp))
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3

ρss = copy(ρ0_p)
steadystate.iterative!(ρss, Hp, Jp)
nss = expect(create(fockbasis)*destroy(fockbasis), ρss)
@test n_an - real(nss) < 1e-3


# Test error messages
#@test_throws ErrorException steadystate.eigenvector(sx, [sm])
function fout_wrong(t, x)
  @assert x == t
end
@test_throws AssertionError steadystate.master(Hdense, Jdense; fout=fout_wrong)

end # testset
