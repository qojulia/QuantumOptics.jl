using Base.Test
using QuantumOptics

@testset "particle" begin

srand(0)

D(op1::Operator, op2::Operator) = abs(tracedistance_nh(full(op1), full(op2)))

N = 200
xmin = -32.5
xmax = 20.1

basis_position = PositionBasis(xmin, xmax, N)
basis_momentum = MomentumBasis(basis_position)

b2 = PositionBasis(basis_momentum)
@test basis_position.xmax - basis_position.xmin ≈ b2.xmax - b2.xmin
@test basis_position.N == b2.N

# Test Gaussian wave-packet in both bases
x0 = 5.1
p0 = -3.2
sigma = 1.
sigma_x = sigma/sqrt(2)
sigma_p = 1./(sigma*sqrt(2))

psi0_bx = gaussianstate(basis_position, x0, p0, sigma)
psi0_bp = gaussianstate(basis_momentum, x0, p0, sigma)

@test 1 ≈ norm(psi0_bx)
@test 1 ≈ norm(psi0_bp)

p_bx = momentum(basis_position)
x_bx = position(basis_position)

@test 1e-10 > D(p_bx, transform(basis_position, basis_momentum)*full(momentum(basis_momentum))*transform(basis_momentum, basis_position))

p_bp = momentum(basis_momentum)
x_bp = position(basis_momentum)

@test x0 ≈ expect(x_bx, psi0_bx)
@test 0.1 > abs(p0 - expect(p_bx, psi0_bx))
@test p0 ≈ expect(p_bp, psi0_bp)
@test 0.1 > abs(x0 - expect(x_bp, psi0_bp))

@test 1e-13 > abs(variance(x_bx, psi0_bx) - sigma^2/2)
@test 1e-13 > abs(variance(x_bp, psi0_bp) - sigma^2/2)
@test 1e-13 > abs(variance(p_bx, psi0_bx) - 1/(2*sigma^2))
@test 1e-13 > abs(variance(p_bp, psi0_bp) - 1/(2*sigma^2))

# Test potentialoperator
V(x) = x^2
V_bx = potentialoperator(basis_position, V)
V_bp = potentialoperator(basis_momentum, V)

@test expect(V_bp, psi0_bp) ≈ expect(V_bx, psi0_bx)


# Test FFT transformation
function transformation(b1::MomentumBasis, b2::PositionBasis, psi::Ket)
    Lp = (b1.pmax - b1.pmin)
    dx = particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dx - Lp)/Lp > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp.(1im*b2.xmin*(particle.samplepoints(b1)-b1.pmin)).*psi.data
    psi_fft = exp.(1im*b1.pmin*particle.samplepoints(b2)).*ifft(psi_shifted)*sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation(b1::PositionBasis, b2::MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp.(-1im*b2.pmin*(particle.samplepoints(b1)-b1.xmin)).*psi.data
    psi_fft = exp.(-1im*b1.xmin*particle.samplepoints(b2)).*fft(psi_shifted)/sqrt(N)
    return Ket(b2, psi_fft)
end

psi0_bx_fft = transformation(basis_position, basis_momentum, psi0_bx)
psi0_bp_fft = transformation(basis_momentum, basis_position, psi0_bp)

@test 0.1 > abs(x0 - expect(x_bp, psi0_bx_fft))
@test p0 ≈ expect(p_bp, psi0_bx_fft)
@test 0.1 > abs(p0 - expect(p_bx, psi0_bp_fft))
@test x0 ≈ expect(x_bx, psi0_bp_fft)

@test 1e-12 > norm(psi0_bx_fft - psi0_bp)
@test 1e-12 > norm(psi0_bp_fft - psi0_bx)


Tpx = transform(basis_momentum, basis_position)
Txp = transform(basis_position, basis_momentum)
psi0_bx_fft = Tpx*psi0_bx
psi0_bp_fft = Txp*psi0_bp

@test 0.1 > abs(x0 - expect(x_bp, psi0_bx_fft))
@test p0 ≈ expect(p_bp, psi0_bx_fft)
@test 0.1 > abs(p0 - expect(p_bx, psi0_bp_fft))
@test x0 ≈ expect(x_bx, psi0_bp_fft)

@test 1e-12 > norm(psi0_bx_fft - psi0_bp)
@test 1e-12 > norm(psi0_bp_fft - psi0_bx)

@test 1e-12 > norm(dagger(Tpx*psi0_bx) - dagger(psi0_bx)*dagger(Tpx))
@test 1e-12 > norm(dagger(Txp*psi0_bp) - dagger(psi0_bp)*dagger(Txp))

# Test gemv!
psi_ = deepcopy(psi0_bp)
operators.gemv!(Complex(1.), Tpx, psi0_bx, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psi0_bp)
operators.gemv!(Complex(1.), Tpx, psi0_bx, Complex(1.), psi_)
@test 1e-12 > norm(psi_ - 2*psi0_bp)

psi_ = deepcopy(psi0_bx)
operators.gemv!(Complex(1.), Txp, psi0_bp, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psi0_bx)
operators.gemv!(Complex(1.), Txp, psi0_bp, Complex(1.), psi_)
@test 1e-12 > norm(psi_ - 2*psi0_bx)


alpha = complex(3.2)
beta = complex(-1.2)
randdata1 = rand(Complex128, N)
randdata2 = rand(Complex128, N)

state = Ket(basis_position, randdata1)
result_ = Ket(basis_momentum, copy(randdata2))
result0 = alpha*full(Tpx)*state + beta*result_
operators.gemv!(alpha, Tpx, state, beta, result_)
@test 1e-11 > norm(result0 - result_)

state = Bra(basis_position, randdata1)
result_ = Bra(basis_momentum, copy(randdata2))
result0 = alpha*state*full(Txp) + beta*result_
operators.gemv!(alpha, state, Txp, beta, result_)
@test 1e-11 > norm(result0 - result_)

state = Ket(basis_momentum, randdata1)
result_ = Ket(basis_position, copy(randdata2))
result0 = alpha*full(Txp)*state + beta*result_
operators.gemv!(alpha, Txp, state, beta, result_)
@test 1e-11 > norm(result0 - result_)

state = Bra(basis_momentum, randdata1)
result_ = Bra(basis_position, copy(randdata2))
result0 = alpha*state*full(Tpx) + beta*result_
operators.gemv!(alpha, state, Tpx, beta, result_)
@test 1e-11 > norm(result0 - result_)


# Test gemm!
rho0_xx = tensor(psi0_bx, dagger(psi0_bx))
rho0_xp = tensor(psi0_bx, dagger(psi0_bp))
rho0_px = tensor(psi0_bp, dagger(psi0_bx))
rho0_pp = tensor(psi0_bp, dagger(psi0_bp))

rho_ = DenseOperator(basis_momentum, basis_position)
operators.gemm!(Complex(1.), Tpx, rho0_xx, Complex(0.), rho_)
@test 1e-12 > D(rho_, rho0_px)
@test 1e-12 > D(Tpx*rho0_xx, rho0_px)

rho_ = DenseOperator(basis_position, basis_momentum)
operators.gemm!(Complex(1.), rho0_xx, Txp, Complex(0.), rho_)
@test 1e-12 > D(rho_, rho0_xp)
@test 1e-12 > D(rho0_xx*Txp, rho0_xp)

rho_ = DenseOperator(basis_momentum, basis_momentum)
operators.gemm!(Complex(1.), Tpx, rho0_xp, Complex(0.), rho_)
@test 1e-12 > D(rho_, rho0_pp)
@test 1e-12 > D(Tpx*rho0_xx*Txp, rho0_pp)

rho_ = DenseOperator(basis_momentum, basis_momentum)
operators.gemm!(Complex(1.), rho0_px, Txp, Complex(0.), rho_)
@test 1e-12 > D(rho_, rho0_pp)
@test 1e-12 > D(Txp*rho0_pp*Tpx, rho0_xx)


alpha = complex(3.2)
beta = complex(-1.2)
randdata1 = rand(Complex128, N, N)
randdata2 = rand(Complex128, N, N)

op = DenseOperator(basis_position, basis_position, randdata1)
result_ = DenseOperator(basis_momentum, basis_position, copy(randdata2))
result0 = alpha*full(Tpx)*op + beta*result_
operators.gemm!(alpha, Tpx, op, beta, result_)
@test 1e-11 > D(result0, result_)

result_ = DenseOperator(basis_position, basis_momentum, copy(randdata2))
result0 = alpha*op*full(Txp) + beta*result_
operators.gemm!(alpha, op, Txp, beta, result_)
@test 1e-11 > D(result0, result_)

op = DenseOperator(basis_momentum, basis_momentum, randdata1)
result_ = DenseOperator(basis_position, basis_momentum, copy(randdata2))
result0 = alpha*full(Txp)*op + beta*result_
operators.gemm!(alpha, Txp, op, beta, result_)
@test 1e-11 > D(result0, result_)

result_ = DenseOperator(basis_momentum, basis_position, copy(randdata2))
result0 = alpha*op*full(Tpx) + beta*result_
operators.gemm!(alpha, op, Tpx, beta, result_)
@test 1e-11 > D(result0, result_)



# Test FFT with lazy product
psi_ = deepcopy(psi0_bx)
operators.gemv!(Complex(1.), LazyProduct(Txp, Tpx), psi0_bx, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psi0_bx)
@test 1e-12 > norm(Txp*(Tpx*psi0_bx) - psi0_bx)

psi_ = deepcopy(psi0_bx)
I = full(identityoperator(basis_momentum))
operators.gemv!(Complex(1.), LazyProduct(Txp, I, Tpx), psi0_bx, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psi0_bx)
@test 1e-12 > norm(Txp*I*(Tpx*psi0_bx) - psi0_bx)

# Test dense FFT operator
Txp_dense = DenseOperator(Txp)
Tpx_dense = DenseOperator(Tpx)
@test isa(Txp_dense, DenseOperator)
@test isa(Tpx_dense, DenseOperator)
@test 1e-5 > D(Txp_dense*rho0_pp*Tpx_dense, rho0_xx)

# Test FFT in 2D
N = [50, 40]
xmin = [-32.5, -10π]
xmax = [24.1, 9π]

basis_position = [PositionBasis(xmin[i], xmax[i], N[i]) for i=1:2]
basis_momentum = MomentumBasis.(basis_position)

x0 = [5.1, -0.2]
p0 = [-3.2, 1.33]
sigma = [1., 0.9]
sigma_x = sigma./sqrt(2)
sigma_p = 1./(sigma.*sqrt(2))

Txp = transform(tensor(basis_position...), tensor(basis_momentum...))
Tpx = transform(tensor(basis_momentum...), tensor(basis_position...))

Txp_sub = [transform(basis_position[i], basis_momentum[i]) for i=1:2]
Tpx_sub = dagger.(Txp_sub)
Txp_full = full.(Txp_sub)
Txp_comp = tensor(Txp_full...)

difference = (full(Txp) - Txp_comp).data
@test isapprox(difference, zeros(difference); atol=1e-12)

psi0_x = gaussianstate.(basis_position, x0, p0, sigma_x)
psi0_p = gaussianstate.(basis_momentum, x0, p0, sigma_p)

psi_p_fft = Tpx*tensor(psi0_x...)
psi_p_fft2 = tensor((Tpx_sub.*psi0_x)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_x_fft = Txp*tensor(psi0_p...)
psi_x_fft2 = tensor((Txp_sub.*psi0_p)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_p_fft = dagger(tensor(psi0_x...))*Txp
psi_p_fft2 = tensor((dagger.(psi0_x).*Txp_sub)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_x_fft = dagger(tensor(psi0_p...))*Tpx
psi_x_fft2 = tensor((dagger.(psi0_p).*Tpx_sub)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

difference = (full(Txp) - identityoperator(DenseOperator, Txp.basis_l)*Txp).data
@test isapprox(difference, zeros(difference); atol=1e-12)
@test_throws AssertionError transform(tensor(basis_position...), tensor(basis_position...))
@test_throws particle.IncompatibleBases transform(SpinBasis(1//2)^2, SpinBasis(1//2)^2)

@test full(Txp) == full(Txp_sub[1] ⊗ Txp_sub[2])

# Test ket only FFTs
Txp = transform(tensor(basis_position...), tensor(basis_momentum...); ket_only=true)
Tpx = transform(tensor(basis_momentum...), tensor(basis_position...); ket_only=true)

Txp_sub = [transform(basis_position[i], basis_momentum[i]; ket_only=true) for i=1:2]
Tpx_sub = dagger.(Txp_sub)

psi_p_fft = Tpx*tensor(psi0_x...)
psi_p_fft2 = tensor((Tpx_sub.*psi0_x)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_x_fft = Txp*tensor(psi0_p...)
psi_x_fft2 = tensor((Txp_sub.*psi0_p)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_p_fft = dagger(tensor(psi0_x...))*Txp
psi_p_fft2 = tensor((dagger.(psi0_x).*Txp_sub)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

psi_x_fft = dagger(tensor(psi0_p...))*Tpx
psi_x_fft2 = tensor((dagger.(psi0_p).*Tpx_sub)...)
@test norm(psi_p_fft - psi_p_fft2) < 1e-15

end # testset
