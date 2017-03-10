using Base.Test
using QuantumOptics

@testset "superoperators" begin

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
ρ₀ = Ψ₀⊗dagger(Ψ₀)


op1 = DenseOperator(spinbasis, [1.2+0.3im 0.7+1.2im;0.3+0.1im 0.8+3.2im])
op2 = DenseOperator(spinbasis, [0.2+0.1im 0.1+2.3im; 0.8+4.0im 0.3+1.4im])
@test tracedistance(spre(op1)*op2, op1*op2) < 1e-12
@test tracedistance(spost(op1)*op2, op2*op1) < 1e-12

@test spre(SparseOperator(op1))*op2 == op1*op2
@test spost(SparseOperator(op1))*op2 == op2*op1

L = liouvillian(H, J)
ρ = -1im*(H*ρ₀ - ρ₀*H)
for j=J
    ρ += j*ρ₀*dagger(j) - 0.5*dagger(j)*j*ρ₀ - 0.5*ρ₀*dagger(j)*j
end
@test tracedistance(L*ρ₀, ρ) < 1e-10

tout, ρt = timeevolution.master([0.,1.], ρ₀, H, J; reltol=1e-7)
@test tracedistance(expm(full(L))*ρ₀, ρt[end]) < 1e-6

@test_throws DimensionMismatch DenseSuperOperator(L.basis_l, L.basis_r, full(L).data[1:end-1, 1:end-1])
@test_throws DimensionMismatch SparseSuperOperator(L.basis_l, L.basis_r, L.data[1:end-1, 1:end-1])

@test full(spre(op1)) == spre(op1)

@test_throws DimensionMismatch L*op1
@test_throws DimensionMismatch L*spre(sm)

@test L/2.0 == 0.5*L == L*0.5
@test -L == SparseSuperOperator(L.basis_l, L.basis_r, -L.data)

@test_throws AssertionError liouvillian(H, J; Gamma=zeros(4, 4))

Gamma = diagm([1.0, 1.0])
@test liouvillian(H, J; Gamma=Gamma) == L

end # testset
