using Test
using QuantumOptics
using LinearAlgebra

@testset "semiclassical" begin

# Test equality
b = GenericBasis(3)
x1 = semiclassical.State(basisstate(b, 1), [0.5im])
x2 = semiclassical.State(basisstate(b, 1), [0.6im])
x3 = semiclassical.State(basisstate(b, 2), [0.5im])

@test x1 == x1
@test x1 != x2
@test x1 != x3

x1 = semiclassical.State(dm(basisstate(b, 1)), [0.5im])
x2 = semiclassical.State(dm(basisstate(b, 1)), [0.6im])
x3 = semiclassical.State(dm(basisstate(b, 2)), [0.5im])

@test x1 == x1
@test x1 != x2
@test x1 != x3


# Test expect and variance
b = FockBasis(10)
a = destroy(b)
n = number(b)

alpha = complex(0.4, 0.3)
psi = coherentstate(b, alpha)
rho = dm(psi)

x_ket = semiclassical.State(psi, [complex(1., 0.)])
x_rho = semiclassical.State(rho, [complex(1., 0.)])

@test expect(a, x_ket) ≈ alpha
@test expect(a, x_rho) ≈ alpha
@test expect(a, [x_ket, x_rho]) ≈ [alpha, alpha]
@test variance(n, x_ket) ≈ abs2(alpha)
@test variance(n, x_rho) ≈ abs2(alpha)
@test variance(n, [x_ket, x_rho]) ≈ [abs2(alpha), abs2(alpha)]

# Test partial tr
b1 = GenericBasis(3)
b2 = GenericBasis(5)
b = b1 ⊗ b2
psi = randstate(b)
x = semiclassical.State(psi, [0.4, -0.3im])
@test ptrace(x, 1).quantum == ptrace(psi, 1)
@test ptrace(x, [2]).quantum == ptrace(psi, 2)

rho = randoperator(b)
x = semiclassical.State(rho, [0.4, -0.3im])
@test ptrace(x, 1).quantum == ptrace(rho, 1)
@test ptrace(x, [2]).quantum == ptrace(rho, 2)

# Test dm function
b = GenericBasis(4)
psi = randstate(b)
u = ComplexF64[complex(2., 3.)]
state = semiclassical.State(psi, u)
@test dm(state) == semiclassical.State(dm(psi), u)


# Test casting between and semiclassical states and complex vectors
b = GenericBasis(4)
rho = randoperator(b)
u = rand(ComplexF64, 3)
state = semiclassical.State(rho, u)
state_ = semiclassical.State(randoperator(b), rand(ComplexF64, 3))
x = Vector{ComplexF64}(undef, 19)
semiclassical.recast!(state, x)
semiclassical.recast!(x, state_)
@test state_ == state


# Test master
spinbasis = SpinBasis(1//2)

# Random 2 level Hamiltonian
a1 = 0.5
a2 = 1.9
c = 1.3
d = -4.7

data = [a1 c-1im*d; c+1im*d a2]
H = DenseOperator(spinbasis, data)

a = (a1 + a2)/2
b = (a1 - a2)/2
r = [c d b]

sigma_r = c*sigmax(spinbasis) + d*sigmay(spinbasis) + b*sigmaz(spinbasis)

U(t) = exp(-1im*a*t)*(cos(norm(r)*t)*one(spinbasis) - 1im*sin(norm(r)*t)*sigma_r/norm(r))

# Random initial state
psi0 = randstate(spinbasis)


T = [0:0.5:1;]

fquantum_schroedinger(t, rho, u) = H
fquantum_master(t, rho, u) = H, [], []
function fclassical(t, quantumstate, u, du)
    du[1] = -1*u[1]
end

state0 = semiclassical.State(psi0, ComplexF64[complex(2., 3.)])
function f(t, state::semiclassical.State{B,T}) where {B<:Basis,T<:Ket{B}}
    @test 1e-5 > norm(state.quantum - U(t)*psi0)
    @test 1e-5 > abs(state.classical[1] - state0.classical[1]*exp(-t))
end
semiclassical.schroedinger_dynamic(T, state0, fquantum_schroedinger, fclassical; fout=f)
tout, state_t = semiclassical.schroedinger_dynamic(T, state0, fquantum_schroedinger, fclassical)
f(T[end], state_t[end])

function f(t, state::semiclassical.State{B,T}) where {B<:Basis,T<:DenseOperator{B,B}}
    @test 1e-5 > tracedistance(state.quantum, dm(U(t)*psi0))
    @test 1e-5 > abs(state.classical[1] - state0.classical[1]*exp(-t))
end
semiclassical.master_dynamic(T, state0, fquantum_master, fclassical; fout=f)
tout, state_t = semiclassical.master_dynamic(T, state0, fquantum_master, fclassical)
f(T[end], state_t[end])

# Test mcwf
# Set up system where only atom can jump once
ba = SpinBasis(1//2)
bf = FockBasis(5)
sm = sigmam(ba)⊗one(bf)
a = one(ba)⊗destroy(bf)
H = 0*sm
J = [0*a,sm]
Jdagger = dagger.(J)
function fquantum(t,psi,u)
    return H, J, Jdagger
end
function fclassical(t,psi,u,du)
    du[1] = u[2] # dx
    du[2] = 0.0
end
njumps = [0]
function fjump_classical(t,psi,u,i)
    @test i==2
    njumps .+= 1
    u[2] += 1.0
end
u0 = rand(2) .+ 0.0im
ψ0 = semiclassical.State(spinup(ba)⊗fockstate(bf,0),u0)

tout1, ψt1 = semiclassical.mcwf_dynamic(T,ψ0,fquantum,fclassical,fjump_classical,seed=1)
@test njumps == [1]
tout2, ψt2 = semiclassical.mcwf_dynamic(T,ψ0,fquantum,fclassical,fjump_classical,seed=1)
@test ψt2 == ψt1
tout3, ψt3 = semiclassical.mcwf_dynamic(T,ψ0,fquantum,fclassical,fjump_classical;display_beforeevent=true,seed=1)
@test length(ψt3) == length(ψt1)+1
tout4, ψt4 = semiclassical.mcwf_dynamic(T,ψ0,fquantum,fclassical,fjump_classical;display_beforeevent=true,display_afterevent=true,seed=1)
@test length(ψt4) == length(ψt1)+2
tout5, ut = semiclassical.mcwf_dynamic(T,ψ0,fquantum,fclassical,fjump_classical;display_beforeevent=true,display_afterevent=true,seed=1,fout=(t,psi)->copy(psi.classical))

@test ψt1[end].classical[2] == u0[2] + 1.0

# Test classical jump behavior
before_jump = findfirst(t -> !(t∈T), tout3)
after_jump = findlast(t-> !(t∈T), tout4)
@test after_jump == before_jump+1
@test ψt3[before_jump].classical[2] == u0[2]
@test ψt4[after_jump].classical[2] == u0[2] + 1.0
@test ut == [ψ.classical for ψ=ψt4]

# Test quantum jumps
@test ψt1[end].quantum == spindown(ba)⊗fockstate(bf,0)
@test ψt4[before_jump].quantum == ψ0.quantum
@test ψt4[after_jump].quantum == spindown(ba)⊗fockstate(bf,0)

end # testsets
