using Base.Test
using QuantumOptics

ωc = 1.2
ωa = 0.9
g = 1.0
Ncutoff = 10

T = Float64[0.,1.]


fockbasis = FockBasis(Ncutoff)
spinbasis = SpinBasis(1//2)
basis = tensor(spinbasis, fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

Ha = embed(basis, 1, 0.5*ωa*sz)
Hc = embed(basis, 2, ωc*number(fockbasis))
Hint = g/2*(sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis))

H = Ha + Hc + Hint

Omega_n(n::Int) = sqrt((ωa-ωc)^2 + g^2*(n+1))
Alpha_n(n::Int) = atan2(g*sqrt(n+1),(ωa-ωc))

Em(n::Int) = ωc*(n+0.5) - 0.5*Omega_n(n)
Ep(n::Int) = ωc*(n+0.5) + 0.5*Omega_n(n)

psi_e(n::Int) = spinup(spinbasis) ⊗ fockstate(fockbasis, n)
psi_g(n::Int) = spindown(spinbasis) ⊗ fockstate(fockbasis, n+1)

psi_p(n::Int) = cos(Alpha_n(n)/2)*psi_e(n) + sin(Alpha_n(n)/2)*psi_g(n)
psi_m(n::Int) = -sin(Alpha_n(n)/2)*psi_e(n) + cos(Alpha_n(n)/2)*psi_g(n)
basisstates_p = Ket[psi_p(n) for n=0:Ncutoff-1]
basisstates_m = Ket[psi_m(n) for n=0:Ncutoff-1]

Emvec = Float64[Em(n) for n=0:Ncutoff-1]
Epvec = Float64[Ep(n) for n=-1:Ncutoff-1]

P = sortperm([Emvec; Epvec])
basisstates = Ket[basisstates_m; psi_g(-1); basisstates_p][P][1:(Ncutoff+1)]
E = [Emvec; Epvec][P][1:(Ncutoff+1)]

@test norm(operatorspectrum(full(H))[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum_hermitian(full(H))[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum(H)[1:(Ncutoff+1)] - E) < 1e-12
@test norm(operatorspectrum_hermitian(H)[1:(Ncutoff+1)] - E) < 1e-12

b = eigenstates(full(H))[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenstates_hermitian(full(H))[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenstates(H)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

b = eigenstates_hermitian(H)[1:(Ncutoff+1)]
for i=1:length(b)
    @test 1-abs(dagger(b[i])*basisstates[i])<1e-12
end

twospinbasis = spinbasis ⊗ spinbasis
Sx = full(sum([embed(twospinbasis, i, sx) for i=1:2]))/2.
Sy = full(sum([embed(twospinbasis, i, sy) for i=1:2]))/2.
Sz = full(sum([embed(twospinbasis, i, sz) for i=1:2]))/2.
Ssq = Sx^2 + Sy^2 + Sz^2
d, v = simdiag([Sz, Ssq])
@test dz[1] == [-1.0, 0, 0, 1.0]
@test isapprox(d[2],  [2, 0.0, 2, 2])
@test_throws ErrorException simdiag([Sx, Sy])

threespinbasis = spinbasis ⊗ spinbasis ⊗ spinbasis
Sx3 = sum([embed(threespinbasis, i, sx) for i=1:3])/2.
Sy3 = sum([embed(threespinbasis, i, sy) for i=1:3])/2.
Sz3 = sum([embed(threespinbasis, i, sz) for i=1:3])/2.
Ssq3 = Sx3^2 + Sy3^2 + Sz3^2
d3, v3 = simdiag([Ssq3, Sz3])
dsq3_std = eigvals(full(Ssq3).data)
@test isapprox(diagm(dsq3_std), v3'*Ssq3.data*v3)
@test_throws ErrorException simdiag([Sy3, Sz3])
@test_throws ErrorException simdiag([destroy(fockbasis), create(fockbasis)])
