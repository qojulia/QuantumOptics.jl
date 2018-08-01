using Test
using QuantumOptics
using Random, LinearAlgebra

@testset "mcwf" begin

# Define parameters for spin coupled to electric field mode.
ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

Ntrajectories = 500
T = Float64[0.:0.1:10.;]

# Define operators
fockbasis = FockBasis(8)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

# Hamiltonian
Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
H = Ha + Hc + Hint
Hdense = dense(H)
Hlazy = LazySum(Ha, Hc, Hint)

# Jump operators
Ja = embed(basis, 1, sqrt(γ)*sm)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]
Jdense = map(dense, J)
Jlazy = [LazyTensor(basis, 1, sqrt(γ)*sm), LazyTensor(basis, 2, sqrt(κ)*destroy(fockbasis))]

# Initial conditions
Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)


# Test mcwf
tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hdense, Jdense; seed=UInt(1), reltol=1e-7)
tout2, Ψt2 = timeevolution.mcwf(T, Ψ₀, Hdense, Jdense; seed=UInt(1), reltol=1e-7)
@test Ψt == Ψt2
Ψ = Ψt[end]

tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(2), reltol=1e-6)
@test norm(Ψt[end]-Ψ) > 0.1

t_fout = Float64[]
Ψ_fout = []
function fout(t, x)
  push!(t_fout, t)
  push!(Ψ_fout, normalize(x))
end
timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(2), reltol=1e-6, fout=fout)
@test tout == t_fout && Ψt == Ψ_fout


# Test mcwf for irreducible input
tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hlazy, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end] - Ψ) < 1e-5

tout, Ψt2 = timeevolution.mcwf(T, Ψ₀, H, Jlazy; seed=UInt(1), reltol=1e-6, display_beforeevent=true, display_afterevent=true)
@test norm(Ψt2[end] - Ψ) < 1e-5
@test length(Ψt2) > length(Ψt)

tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, Jlazy./[sqrt(γ), sqrt(κ)]; seed=UInt(1), rates=[γ, κ], reltol=1e-6)
@test norm(Ψt[end] - Ψ) < 1e-5

# Test mcwf_h
tout, Ψt = timeevolution.mcwf_h(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_h(T, Ψ₀, H, Jdense; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_h(T, Ψ₀, Hdense, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_h(T, Ψ₀, H, J; seed=UInt(2), reltol=1e-6)
@test norm(Ψt[end]-Ψ) > 0.1


# Test mcwf nh
Hnh = H - 0.5im*sum([dagger(J[i])*J[i] for i=1:length(J)])
Hnh_dense = dense(Hnh)

tout, Ψt = timeevolution.mcwf_nh(T, Ψ₀, Hnh, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_nh(T, Ψ₀, Hnh, Jdense; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_nh(T, Ψ₀, Hnh_dense, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf_nh(T, Ψ₀, Hnh, J; seed=UInt(2), reltol=1e-6)
@test norm(Ψt[end]-Ψ) > 0.1



# Test convergence to master solution
tout_master, ρt_master = timeevolution.master(T, ρ₀, H, J)

ρ_average = DenseOperator[0 * ρ₀ for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hdense, Jdense; seed=UInt(i))
    for j=1:length(T)
        ρ_average[j] += (Ψt[j] ⊗ dagger(Ψt[j]))/Ntrajectories
    end
end
for i=1:length(T)
    err = tracedistance(ρt_master[i], ρ_average[i])
    @test err < 0.1
end


# Test single jump operator
J1 = [Ja]
J1_dense = map(dense, J1)
J2 = [Ja, 0 * Jc]
J2_dense = map(dense, J2)

tout_master, ρt_master = timeevolution.master(T, ρ₀, Hdense, J1_dense)

ρ_average_1 = DenseOperator[0 * ρ₀ for i=1:length(T)]
ρ_average_2 = DenseOperator[0 * ρ₀ for i=1:length(T)]
ρ_average_3 = DenseOperator[0 * ρ₀ for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt_1 = timeevolution.mcwf(T, Ψ₀, Hdense, J1_dense; seed=UInt(i))
    tout, Ψt_2 = timeevolution.mcwf(T, Ψ₀, Hdense, J2_dense; seed=UInt(i))
    tout, Ψt_3 = timeevolution.mcwf(T, Ψ₀, Hdense, [J1_dense[1]/sqrt(γ)]; seed=UInt(i), rates=[γ])
    for j=1:length(T)
        ρ_average_1[j] += (Ψt_1[j] ⊗ dagger(Ψt_1[j]))/Ntrajectories
        ρ_average_2[j] += (Ψt_2[j] ⊗ dagger(Ψt_2[j]))/Ntrajectories
        ρ_average_3[j] += (Ψt_3[j] ⊗ dagger(Ψt_3[j]))/Ntrajectories
    end
end
for i=1:length(T)
    @test tracedistance(ρt_master[i], ρ_average_1[i]) < 0.1
    @test tracedistance(ρt_master[i], ρ_average_2[i]) < 0.1
    @test tracedistance(ρt_master[i], ρ_average_3[i]) < 0.1
end


# Test equivalence to schroedinger time evolution for no decay
J = DenseOperator[]
tout_schroedinger, Ψt_schroedinger = timeevolution.schroedinger(T, Ψ₀, Hdense)
tout_mcwf, Ψt_mcwf = timeevolution.mcwf(T, Ψ₀, Hdense, J)
tout_mcwf_h, Ψt_mcwf_h = timeevolution.mcwf_h(T, Ψ₀, Hdense, J)
tout_mcwf_nh, Ψt_mcwf_nh = timeevolution.mcwf_nh(T, Ψ₀, Hdense, J)

for i=1:length(T)
    @test norm(Ψt_mcwf[i] - Ψt_schroedinger[i]) < 1e-4
    @test norm(Ψt_mcwf_h[i] - Ψt_schroedinger[i]) < 1e-4
    @test norm(Ψt_mcwf_nh[i] - Ψt_schroedinger[i]) < 1e-4
end


# Test diagonal jump operators
threespinbasis = spinbasis ⊗ spinbasis ⊗ spinbasis
Γ, γ1, γ2, = 1.0, 1/sqrt(2), 1/sqrt(3)
rates = [Γ γ1 γ2; γ1 Γ γ1; γ2 γ1 Γ]
J3 = [embed(threespinbasis, i, sm) for i=1:3]
H = sum(J3) + dagger(sum(J3))
d, diagJ = diagonaljumps(rates, J3)
ψ3 = spindown(spinbasis) ⊗ spindown(spinbasis) ⊗ spindown(spinbasis)
tout, ρ3_nondiag = timeevolution.master(T, ψ3, H, J3; rates=rates)
tout, ρ3_diag = timeevolution.master(T, ψ3, H, diagJ; rates=d)

ρ3_avg = DenseOperator[0*ρ3_diag[1] for i=1:length(T)]
for i=1:Ntrajectories
    tout, ψ3t = timeevolution.mcwf(T, ψ3, H, diagJ; rates=d)
    for j=1:length(T)
        ρ3_avg[j] += (ψ3t[j] ⊗ dagger(ψ3t[j]))/Ntrajectories
    end
end

dist = []
for i=1:length(tout)
  @test tracedistance(ρ3_nondiag[i], ρ3_diag[i]) < 1e-14
  @test tracedistance(ρ3_avg[i], ρ3_diag[i]) < 0.1
end

@test_throws ArgumentError timeevolution.mcwf(T, ψ3, H, J3; rates=rates)

J3_lazy = [LazyTensor(threespinbasis, i, sm) for i=1:3]
d, diagJ_lazy = diagonaljumps(rates, J3_lazy)
for i=1:3
    @test dense(diagJ_lazy[i]) == dense(diagJ[i])
end

# Test dynamic
H = sp + sm
J = [sm]
Jdagger = dagger.(J)
function Ht(t, psi)
    H*exp(-(5-t)^2), J, Jdagger
end
function Ht2(t, psi)
    H*exp(-(5-t)^2), J, Jdagger, [1.0]
end
ψ0 = spindown(spinbasis)
ρ0 = dm(ψ0)
tout_master, ρt_master = timeevolution.master_dynamic(T, ρ0, Ht)

ρ_average_1 = DenseOperator[0 * ρ0 for i=1:length(T)]
ρ_average_2 = DenseOperator[0 * ρ0 for i=1:length(T)]
ρ_average_3 = DenseOperator[0 * ρ0 for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt_1 = timeevolution.mcwf_dynamic(T, ψ0, Ht; seed=UInt(i))
    tout, Ψt_2 = timeevolution.mcwf_dynamic(T, ψ0, Ht; seed=UInt(i))
    tout, Ψt_3 = timeevolution.mcwf_dynamic(T, ψ0, Ht2; seed=UInt(i))
    for j=1:length(T)
        ρ_average_1[j] += (Ψt_1[j] ⊗ dagger(Ψt_1[j]))/Ntrajectories
        ρ_average_2[j] += (Ψt_2[j] ⊗ dagger(Ψt_2[j]))/Ntrajectories
        ρ_average_3[j] += (Ψt_3[j] ⊗ dagger(Ψt_3[j]))/Ntrajectories
    end
end

# Test non-hermitian dynamic
H_nh = -0.5im*sum(Jdagger.*J)
function Ht_nh(t, psi)
    H*exp(-(5-t)^2) + H_nh, J, Jdagger
end

ρ_average_4 = DenseOperator[0 * ρ0 for i=1:length(T)]
for i=1:Ntrajectories
    tout, Ψt_4 = timeevolution.mcwf_nh_dynamic(T, ψ0, Ht_nh; seed=UInt(i))
    for j=1:length(T)
        ρ_average_4[j] += (Ψt_4[j] ⊗ dagger(Ψt_4[j]))/Ntrajectories
    end
end
for i=1:length(T)
    @test tracedistance(ρt_master[i], ρ_average_1[i]) < 0.1
    @test tracedistance(ρt_master[i], ρ_average_2[i]) < 0.1
    @test tracedistance(ρt_master[i], ρ_average_3[i]) < 0.1
    @test tracedistance(ρt_master[i], ρ_average_4[i]) < 0.1
end


end # testset
