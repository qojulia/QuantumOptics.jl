using Base.Test
using QuantumOptics

type test_spectralanalysis <: Operator; end

@testset "spectralanalysis" begin

srand(0)

randop(b) = DenseOperator(b, rand(Complex128, length(b), length(b)))
sprandop(b) = sparse(DenseOperator(b, rand(Complex128, length(b), length(b))))

# Test diagonalization
b = GenericBasis(5)
op1 = randop(b)
op2 = DenseOperator(b, Hermitian(rand(5, 5)))
op3 = sprandop(b)
op4 = 0.5*(op3 + dagger(op3))
@test eig(op1)[1] == eigvals!(op1)
@test eigvals(op2) ≈ eig(op2)[1]
@test eig(op2, 1:3) == eig(op2, 1:3)
@test eigs(op3; nev=3)[1] ≈ sort(eigs(op3.data; nev=3)[1], by=abs)
@test eigs(op4; nev=3)[1] ≈ eigs(Hermitian(op4.data); nev=3)[1]

@test_throws ArgumentError eig(test_spectralanalysis())
@test_throws ArgumentError eigs(test_spectralanalysis())
@test_throws ArgumentError eigvals(test_spectralanalysis())
@test_throws ArgumentError eigvals!(test_spectralanalysis())
@test_throws ArgumentError eigvals(op3)

# Test simdiag
spinbasis = SpinBasis(1//2)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
twospinbasis = spinbasis ⊗ spinbasis
Sx = full(sum([embed(twospinbasis, i, sx) for i=1:2]))/2.
Sy = full(sum([embed(twospinbasis, i, sy) for i=1:2]))/2.
Sz = full(sum([embed(twospinbasis, i, sz) for i=1:2]))/2.
Ssq = Sx^2 + Sy^2 + Sz^2
d, v = simdiag([Sz, Ssq])
@test d[1] == [-1.0, 0, 0, 1.0]
@test d[2] ≈ [2, 0.0, 2, 2]
@test_throws ErrorException simdiag([Sx, Sy])

threespinbasis = spinbasis ⊗ spinbasis ⊗ spinbasis
Sx3 = full(sum([embed(threespinbasis, i, sx) for i=1:3])/2.)
Sy3 = full(sum([embed(threespinbasis, i, sy) for i=1:3])/2.)
Sz3 = full(sum([embed(threespinbasis, i, sz) for i=1:3])/2.)
Ssq3 = Sx3^2 + Sy3^2 + Sz3^2
d3, v3 = simdiag([Ssq3, Sz3])
dsq3_std = eigvals(full(Ssq3))
@test diagm(dsq3_std) ≈ v3'*Ssq3.data*v3

fockbasis = FockBasis(4)
@test_throws ErrorException simdiag([Sy3, Sz3])
@test_throws ErrorException simdiag([full(destroy(fockbasis)), full(create(fockbasis))])

end # testset
