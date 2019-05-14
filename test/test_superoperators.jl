using Test
using QuantumOptics
using SparseArrays, LinearAlgebra

@testset "superoperators" begin

# Test creation
b = GenericBasis(3)
@test_throws DimensionMismatch DenseSuperOperator((b, b), (b, b), zeros(ComplexF64, 3, 3))
@test_throws DimensionMismatch SparseSuperOperator((b, b), (b, b), spzeros(ComplexF64, 3, 3))

# Test copy, sparse and dense
b1 = GenericBasis(2)
b2 = GenericBasis(7)
b3 = GenericBasis(5)
b4 = GenericBasis(3)

s = DenseSuperOperator((b1, b2), (b3, b4))
s_ = dense(s)
s_.data[1,1] = 1
@test s.data[1,1] == 0
s_sparse = sparse(s_)
@test isa(s_sparse, SparseSuperOperator)
@test s_sparse.data[1,1] == 1

s = SparseSuperOperator((b1, b2), (b3, b4))
s_ = sparse(s)
s_.data[1,1] = 1
@test s.data[1,1] == 0
s_dense = dense(s_)
@test isa(s_dense, DenseSuperOperator)
@test s_dense.data[1,1] == 1

# Test length
b1 = GenericBasis(3)
b2 = GenericBasis(5)
op = DenseOperator(b1, b2)
S = spre(op)
@test length(S) == length(S.data) == (3*5)^2

# Test arithmetic
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b3 = GenericBasis(5)
b4 = GenericBasis(3)

d1 = DenseSuperOperator((b1, b2), (b3, b4))
d2 = DenseSuperOperator((b3, b4), (b1, b2))
s1 = SparseSuperOperator((b1, b2), (b3, b4))
s2 = SparseSuperOperator((b3, b4), (b1, b2))

x = d1*d2
@test isa(x, DenseSuperOperator)
@test x.basis_l == x.basis_r == (b1, b2)

x = s1*s2
@test isa(x, SparseSuperOperator)
@test x.basis_l == x.basis_r == (b1, b2)

x = d1*s2
@test isa(x, DenseSuperOperator)
@test x.basis_l == x.basis_r == (b1, b2)

x = s1*d2
@test isa(x, DenseSuperOperator)
@test x.basis_l == x.basis_r == (b1, b2)

x = d1*3
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = 2.5*s1
@test isa(x, SparseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = d1 + d1
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = s1 + s1
@test isa(x, SparseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = d1 + s1
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = d1 - d1
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = s1 - s1
@test isa(x, SparseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = d1 - s1
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = -d1
@test isa(x, DenseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

x = -s1
@test isa(x, SparseSuperOperator)
@test x.basis_l == (b1, b2)
@test x.basis_r == (b3, b4)

# Test conversion of unitary matrices to superoperators.
b5 = GenericBasis(4)
CZ = DenseOperator(b5, b5, diagm(0 => [1,1,1,-1]))
sopCZ = SuperOperator(CZ)
@test diag(sopCZ.data) ==  ComplexF64[1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1]
@test sopCZ.basis_l == sopCZ.basis_r == (b5, b5)

# TODO: Clean-up this part
ωc = 1.2
ωa = 0.9
g = 1.0
γ = 0.5
κ = 1.1

T = Float64[0.,1.]


fockbasis = FockBasis(7)
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

Ja = embed(basis, 1, sqrt(γ)*sm)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J = [Ja, Jc]

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
ρ₀ = dm(Ψ₀)


op1 = DenseOperator(spinbasis, [1.2+0.3im 0.7+1.2im;0.3+0.1im 0.8+3.2im])
op2 = DenseOperator(spinbasis, [0.2+0.1im 0.1+2.3im; 0.8+4.0im 0.3+1.4im])
@test tracedistance(spre(op1)*op2, op1*op2) < 1e-12
@test tracedistance(spost(op1)*op2, op2*op1) < 1e-12

@test spre(sparse(op1))*op2 == op1*op2
@test spost(sparse(op1))*op2 == op2*op1

@test spre(sparse(op1))*sparse(op2) == sparse(op1*op2)
@test spost(sparse(op1))*sparse(op2) == sparse(op2*op1)

L = liouvillian(H, J)
ρ = -1im*(H*ρ₀ - ρ₀*H)
for j=J
    ρ .+= j*ρ₀*dagger(j) - 0.5*dagger(j)*j*ρ₀ - 0.5*ρ₀*dagger(j)*j
end
@test tracedistance(L*ρ₀, ρ) < 1e-10

tout, ρt = timeevolution.master([0.,1.], ρ₀, H, J; reltol=1e-7)
@test tracedistance(exp(dense(L))*ρ₀, ρt[end]) < 1e-6

@test dense(spre(op1)) == spre(op1)

@test L/2.0 == 0.5*L == L*0.5
@test -L == SparseSuperOperator(L.basis_l, L.basis_r, -L.data)

@test_throws AssertionError liouvillian(H, J; rates=zeros(4, 4))

rates = diagm(0 => [1.0, 1.0])
@test liouvillian(H, J; rates=rates) == L

# Test broadcasting
@test L .+ L == L + L
Ldense = dense(L)
@test isa(L .+ Ldense, DenseSuperOperator)
L_ = copy(L)
L .+= L
@test L == 2*L_
L .+= Ldense
@test L == 3*L_
Ldense_ = dense(L_)
Ldense .+= Ldense
@test Ldense == 2*Ldense_
Ldense .+= L
@test isa(Ldense, DenseSuperOperator)
@test isapprox(Ldense.data, 5*Ldense_.data)

end # testset
