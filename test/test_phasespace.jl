@testitem "phasespace" tags = [:phasespace] begin
using QuantumOptics
using Random, LinearAlgebra

Random.seed!(0)

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))

# Test quasi-probability functions
b = FockBasis(100)
alpha = complex(0.3, 0.7)
nfock = 3

X = [-2.1:1.5:2;]
Y = [0.13:1.4:3;]
psi_coherent = coherentstate(b, alpha)
rho_coherent = dm(psi_coherent)
psi_fock = fockstate(b, nfock)
rho_fock = dm(psi_fock)

Qpsi_coherent = qfunc(psi_coherent, X, Y)
Qrho_coherent = qfunc(rho_coherent, X, Y)
Qpsi_fock = qfunc(psi_fock, X, Y)
Qrho_fock = qfunc(rho_fock, X, Y)

Wpsi_coherent = wigner(psi_coherent, X, Y)
Wrho_coherent = wigner(rho_coherent, X, Y)
Wpsi_fock = wigner(psi_fock, X, Y)
Wrho_fock = wigner(rho_fock, X, Y)

laguerre3(x) = (-x^3+9x^2-18x+6)/6

for (i,x)=enumerate(X), (j,y)=enumerate(Y)
    beta = 1.0/sqrt(2)*complex(x, y)
    betastate = coherentstate(b, beta)

    q_coherent = 1/pi*exp(-abs2(alpha-beta))
    @test Qpsi_coherent[i, j] ≈ q_coherent
    @test Qrho_coherent[i, j] ≈ q_coherent
    @test qfunc(psi_coherent, beta) ≈ q_coherent
    @test qfunc(rho_coherent, beta) ≈ q_coherent
    @test qfunc(psi_coherent, x, y) ≈ q_coherent
    @test abs2(dagger(betastate) * psi_coherent)/pi ≈ q_coherent
    @test dagger(betastate) * rho_coherent * betastate/pi ≈ q_coherent

    q_fock = 1/pi*exp(-abs2(beta))*abs2(beta)^nfock/factorial(nfock)
    @test Qpsi_fock[i, j] ≈ q_fock
    @test Qrho_fock[i, j] ≈ q_fock
    @test qfunc(psi_fock, beta) ≈ q_fock
    @test qfunc(rho_fock, beta) ≈ q_fock
    @test abs2(dagger(betastate) * psi_fock)/pi ≈ q_fock
    @test dagger(betastate) * rho_fock * betastate/pi ≈ q_fock

    w_coherent = 1/pi*exp(-2*abs2(alpha-beta))
    @test Wpsi_coherent[i, j] ≈ w_coherent
    @test Wrho_coherent[i, j] ≈ w_coherent
    @test wigner(psi_coherent, beta) ≈ w_coherent
    @test wigner(rho_coherent, beta) ≈ w_coherent

    w_fock = 1/pi*(-1)^nfock*laguerre3(4*abs2(beta))*exp(-2*abs2(beta))
    @test Wpsi_fock[i, j] ≈ w_fock
    @test Wrho_fock[i, j] ≈ w_fock
    @test wigner(psi_fock, beta) ≈ w_fock
    @test wigner(rho_fock, beta) ≈ w_fock
end

# Test with offset
alpha = complex(4.3, 2.7)
nfock = 3

X = [0.1:0.1:4;] .+ real(alpha)
Y = [0.13:0.1:3;] .+ imag(alpha)
psi_coherent = coherentstate(b, alpha)
rho_coherent = dm(psi_coherent)
psi_fock = fockstate(b, nfock)
rho_fock = dm(psi_fock)


Qpsi_coherent = qfunc(psi_coherent, X, Y)
Qrho_coherent = qfunc(rho_coherent, X, Y)
Qpsi_fock = qfunc(psi_fock, X, Y)
Qrho_fock = qfunc(rho_fock, X, Y)

Wpsi_coherent = wigner(psi_coherent, X, Y)
Wrho_coherent = wigner(rho_coherent, X, Y)
Wpsi_fock = wigner(psi_fock, X, Y)
Wrho_fock = wigner(rho_fock, X, Y)

b_off = FockBasis(100,3)
psi_coherent_off = coherentstate(b_off, alpha)
rho_coherent_off = dm(psi_coherent_off)
psi_fock_off = fockstate(b_off, nfock)
rho_fock_off = dm(psi_fock)

@test isapprox(qfunc(psi_coherent, alpha), qfunc(psi_coherent_off, alpha))
@test isapprox(Qpsi_coherent, qfunc(psi_coherent_off, X, Y), atol=1e-6)
@test isapprox(Qrho_coherent, qfunc(rho_coherent_off, X, Y), atol=1e-6)
@test isapprox(qfunc(psi_fock, alpha), qfunc(psi_fock_off, alpha))
@test isapprox(Qpsi_fock, qfunc(psi_fock_off, X, Y), atol=1e-6)
@test isapprox(Qrho_fock, qfunc(rho_fock_off, X, Y), atol=1e-6)

@test isapprox(wigner(psi_coherent, alpha), wigner(psi_coherent_off, alpha), atol=1e-6)
@test isapprox(Wpsi_coherent, wigner(psi_coherent_off, X, Y), atol=1e-5, rtol=1e-5)
@test isapprox(Wrho_coherent, wigner(rho_coherent_off, X, Y), atol=1e-5, rtol=1e-5)
@test isapprox(wigner(psi_fock, alpha), wigner(psi_fock_off, alpha), atol=1e-5, rtol=1e-5)
@test isapprox(Wpsi_fock, wigner(psi_fock_off, X, Y), atol=1e-5, rtol=1e-5)
@test isapprox(Wrho_fock, wigner(rho_fock_off, X, Y), atol=1e-5, rtol=1e-5)


# Test qfunc with rand operators
b = FockBasis(50)
psi = randstate(b)
rho = randoperator(b)
X = [-2.1:1.5:2;]
Y = [-0.5:.8:3;]

Qpsi = qfunc(psi, X, Y)
Qrho = qfunc(rho, X, Y)
for (i,x)=enumerate(X), (j,y)=enumerate(Y)
    c = complex(x, y)/sqrt(2)
    state = coherentstate(b, c)
    q_rho = dagger(state) * rho * state/pi
    q_psi = abs2(dagger(state) *psi)/pi
    @test 1e-14 > abs(Qpsi[i,j] - q_psi)
    @test 1e-14 > abs(Qrho[i,j] - q_rho)
    @test 1e-14 > abs(qfunc(psi, c) - q_psi)
    @test 1e-14 > abs(qfunc(rho, c) - q_rho)
end

# Test SU(2) phasespace
b = SpinBasis(5)
theta = π*rand()
phi =2π*rand()
css = coherentspinstate(b,theta,phi)
dmcss = dm(css)
csssx = coherentspinstate(b,π/2,0)
dmcsssx = dm(csssx)
rs = randstate(b)
dmrs = dm(rs)
sx = expect(sigmax(b)/2,css); # eigenstate of jx operator
sy = expect(sigmay(b)/2,css); # eigenstate of jy
sz = expect(sigmaz(b)/2,css); # eigenstate of jz operator
ssq = sx^2 + sy^2 + sz^2

qsu2sx = qfuncsu2(csssx,theta,phi)
qsu2sxdm = qfuncsu2(dmcsssx,theta,phi)
res = 250
costhetam = Array{Float64}(undef,res,2*res)
for i = 1:res, j = 1:2*res
    costhetam[i,j] = sin(i*1pi/(res-1))
end
wsu2 = sum(wignersu2(rs,res).*costhetam)*(π/res)^2
wsu2dm = sum(wignersu2(dmrs,res).*costhetam)*(π/res)^2
qsu2 = sum(qfuncsu2(rs,res).*costhetam)*(π/res)^2
qsu2dm = sum(qfuncsu2(dmrs,res).*costhetam)*(π/res)^2

@test ssq ≈ float(b.spinnumber)^2

@test isapprox(qsu2sxdm, (2*float(b.spinnumber)+1)/(4pi)*(0.5*(sin(theta)cos(phi)+1))^(2*float(b.spinnumber)),atol=1e-2)
@test isapprox(qsu2sx, (2*float(b.spinnumber)+1)/(4pi)*(0.5*(sin(theta)cos(phi)+1))^(2*float(b.spinnumber)),atol=1e-2)
@test isapprox(qsu2, 1.0, atol=1e-2)
@test isapprox(qsu2dm, 1.0, atol=1e-2)

@test isapprox(wignersu2(csssx,π/2,0), (4*float(b.spinnumber)+1)/(4pi), atol=1e-2)
@test isapprox(wignersu2(dmcsssx,π/2,0), (4*float(b.spinnumber)+1)/(4pi),atol=1e-2)
@test isapprox(wsu2, 1.0, atol=1e-2)
@test isapprox(wsu2dm, 1.0, atol=1e-2)


# Test CSS for large spin number (#326)
b = SpinBasis(35)  # S=35 overflows `binomial` function, e.g. binomial(70, 26)
theta = π*rand()
phi =2π*rand()
css = coherentspinstate(b,theta,phi)
css0 = coherentspinstate(b,0,0)
css2 = exp(-0.5im * phi * dense(sigmaz(b))) * exp(-0.5im * theta * dense(sigmay(b))) * css0
@test norm(css - css2) < 1e-12
b_big = SpinBasis(1000)  # > 2000 overflows recursive binomial un-moderated with α^n
css_big = coherentspinstate(b_big,theta,phi)
@test abs(1.0 - norm(css_big)) < 1e-12
# we can go bigger than 10000 without overflowing (≈21000 is the limit), but
# at some point we lose more precision to floating-point errors than we'd
# normally want tolerate.
# @test abs(1 - norm(coherentspinstate(SpinBasis(10000),theta,phi))) < 1e-10

########### YLM test #############
res = 1000
global int = 0
l = rand(4:33)
m = rand(0:l-1)
for i = 0:2pi/res:2pi, j = 0:pi/res:pi
    global int += sin(j)*abs2(ylm(l,m,j,i))
end
t1 = abs(int*2*(pi/res)^2)
@test isapprox(t1, 1.00, atol=1e-2)

l1 = rand(33:40)
m1 = rand(8:30)
l2 = rand(77:80)
m2 = rand(0:10)
global int = 0
for i = 0:2pi/res:2pi, j = 0:pi/res:pi
    global int += sin(j)*ylm(l1,m1,j,i)*conj(ylm(l2,m2,j,i))
end
t2 = abs(int*2*(pi/res)^2)
@test isapprox(t2, 0, atol=1e-2)

end # testset
