using Test
using QuantumOptics

# Implement custom type with AbstractArray interface
mutable struct TestData{T,N,X} <: AbstractArray{T,N}
    x::X
    function TestData(x::X) where X
        x_ = complex.(x)
        new{ComplexF64,length(axes(x)),typeof(x_)}(x_)
    end
end
Base.size(A::TestData) = size(A.x)
Base.getindex(A::TestData, inds...) = getindex(A.x, inds...)
Base.setindex!(A::TestData, val, inds...) = setindex!(A.x, val, inds...)
Base.similar(A::TestData, dims::Int...) = TestData(similar(A.x,dims...))

@testset "abstract-data" begin


###############
# Schrödinger #
N = 3
Ncutoff = 2
T = [0.:0.1:1.;]

Ω = [0 2 3;
     2 0 1;
     3 1 0]

ω = [1., 1.2, 1.5]

basis_fock = FockBasis(Ncutoff)
basis = tensor([basis_fock for i=1:N]...)

a = destroy(basis_fock)
at = create(basis_fock)
I = identityoperator(basis_fock)


psi0 = tensor([coherentstate(basis_fock, i%Ncutoff) for i=1:N]...)


# Interaction picture
Hrot = SparseOpType[]
for i=1:N, j=1:N
    if i==j
        continue
    end
    h = embed(basis, [i,j], [a, Ω[i,j]*at])
    push!(Hrot, h)
end
Hrot = sum(Hrot)

# Schroedinger picture
function f(t, psi)
    H = SparseOpType[embed(basis, i, ω[i]*at*a) for i=1:N]
    for i=1:N, j=1:N
        if i==j
            continue
        end
        h = embed(basis, [i,j], [a, exp(1im*(ω[i]-ω[j])*t)*Ω[i,j]*at])
        push!(H, h)
    end
    sum(H)
end

Hrot_ = Operator(Hrot.basis_l,Hrot.basis_r,TestData(Hrot.data))
tout, psi_rot_t = timeevolution.schroedinger(T, psi0, Hrot)
tout, psi_rot_t_ = timeevolution.schroedinger(T, psi0, Hrot_)
for (psi1,psi2)=zip(psi_rot_t,psi_rot_t_)
    @test psi1==psi2
end

psi0_ = Ket(basis, TestData(psi0.data))
tout, psi_rot_t_ = timeevolution.schroedinger(T, psi0_, Hrot_)
for (psi1,psi2)=zip(psi_rot_t,psi_rot_t_)
    @test psi1.data≈psi2.data
end


##########
# Master #

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
H_ = Ha + Hc + Hint
H = Operator(basis, TestData(H_.data))

Ja_unscaled = embed(basis, 1, sm)
Jc_unscaled = embed(basis, 2, destroy(fockbasis))
Junscaled = [Ja_unscaled, Jc_unscaled]

Ja_ = embed(basis, 1, sqrt(γ)*sm)
Jc_ = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
Ja = Operator(basis, TestData(Ja_.data))
Jc = Operator(basis, TestData(Jc_.data))
J = [Ja, Jc]
Jlazy = [LazyTensor(basis, 1, sqrt(γ)*Operator(spinbasis, TestData(sm.data))), Jc]

Hnh = H - 0.5im*sum([dagger(J[i])*J[i] for i=1:length(J)])

Hdense = dense(H)
Hlazy = LazySum(Ha, Hc, Hint)
Hnh_dense = dense(Hnh)
Junscaled_dense = map(dense, Junscaled)
Jdense = map(dense, J)

Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
ρ₀ = Operator(basis, TestData(dm(Ψ₀).data))

# Test Liouvillian
L = liouvillian(H, J)
ρ = -1im*(H*ρ₀ - ρ₀*H)
for j=J
    ρ .+= j*ρ₀*dagger(j) - 0.5*dagger(j)*j*ρ₀ - 0.5*ρ₀*dagger(j)*j
end
@test tracedistance(L*ρ₀, ρ) < 1e-10

# Test master
tout, ρt = timeevolution.master(T, ρ₀, Hdense, Jdense; reltol=1e-7)
ρ = ρt[end]
@test isa(ρ.data, TestData)
@test tracedistance(dense(exp(dense(L)*T[end])*ρ₀), dense(ρ)) < 1e-6

@test isa(ρ₀.data, TestData) && isa(H.data,TestData) && all(isa(j.data,TestData) for j=J)
tout, ρt = timeevolution.master(T, ρ₀, H, J; reltol=1e-6)
@test isa(ρt[end].data, TestData)
@test tracedistance(dense(ρt[end]), dense(ρ)) < 1e-5


########
# MCWF #

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
H_ = Ha + Hc + Hint
H = Operator(basis, TestData(H_.data))
Hdense = dense(H)

# Jump operators
Ja = embed(basis, 1, sqrt(γ)*sm)
Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
J_ = [Ja, Jc]
J = [Operator(basis, TestData(j.data)) for j=J_]
Jdense = map(dense, J)

# Initial conditions
Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)

# Test mcwf
tout, Ψt = timeevolution.mcwf(T, Ψ₀, Hdense, Jdense; seed=UInt(1), reltol=1e-7)
tout2, Ψt2 = timeevolution.mcwf(T, Ψ₀, Hdense, Jdense; seed=UInt(1), reltol=1e-7)
@test Ψt == Ψt2
Ψ = Ψt[end]

tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(1), reltol=1e-6)
@test norm(Ψt[end]-Ψ) < 1e-5

tout, Ψt = timeevolution.mcwf(T, Ψ₀, H, J; seed=UInt(2), reltol=1e-6)
@test norm(Ψt[end]-Ψ) > 0.1


#####################
# Two-level example #

basis = SpinBasis(1//2)

# Random 2 level Hamiltonian
a1 = 0.5
a2 = 1.9
c = 1.3
d = -4.7

data = [a1 c-1im*d; c+1im*d a2]
H = DenseOperator(basis, data)

a = (a1 + a2)/2
b = (a1 - a2)/2
r = [c d b]

sigma_r = c*sigmax(basis) + d*sigmay(basis) + b*sigmaz(basis)

U(t) = exp(-1im*a*t)*(cos(norm(r)*t)*one(basis) - 1im*sin(norm(r)*t)*sigma_r/norm(r))

# Random initial state
psi0 = randstate(basis)
T = [0:0.5:1;]

f(t, psi::Ket) = @test 1e-5 > norm(psi - U(t)*psi0)
H_ = Operator(basis,TestData(H.data))
timeevolution.schroedinger(T, psi0, H_; fout=f)
timeevolution.mcwf(T, psi0, H_, []; fout=f)

f(t, rho) = @test 1e-5 > tracedistance(dense(rho), dm(U(t)*psi0))
rho0 = Operator(basis, TestData(dm(psi0).data))
timeevolution.master(T, rho0, H_, []; fout=f)

end # testset
