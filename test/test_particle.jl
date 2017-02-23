using Base.Test
using QuantumOptics

@testset "particle" begin

N = 500
xmin = -62.5
xmax = 70.1

basis_position = PositionBasis(xmin, xmax, N)
basis_momentum = MomentumBasis(basis_position)

# Test Gaussian wave-packet in both bases
x0 = 5.1
p0 = -3.2
sigma = 1.
sigma_x = sigma/sqrt(2)
sigma_p = 1./(sigma*sqrt(2))

psix0 = gaussianstate(basis_position, x0, p0, sigma)
psip0 = gaussianstate(basis_momentum, x0, p0, sigma)

@test 1 ≈ norm(psix0)
@test 1 ≈ norm(psip0)

opx_p = momentumoperator(basis_position)
opx_x = positionoperator(basis_position)

opp_p = momentumoperator(basis_momentum)
opp_x = positionoperator(basis_momentum)

@test x0 ≈ expect(opx_x, psix0)
@test 0.1 > abs(p0 - expect(opx_p, psix0))
@test p0 ≈ expect(opp_p, psip0)
@test 0.1 > abs(x0 - expect(opp_x, psip0))

# Test position and momentum operators
@test 0.3 > abs(expect(-opx_p^2, psix0) - expect(laplace_x(basis_position), psix0))
@test expect(opp_p^2, psip0) ≈ expect(laplace_x(basis_momentum), psip0)
@test expect(opx_x^2, psix0) ≈ expect(laplace_p(basis_position), psix0)
@test 0.3 > abs(expect(-opp_x^2, psip0) - expect(laplace_p(basis_momentum), psip0))

# Test FFT transformation
function transformation(b1::MomentumBasis, b2::PositionBasis, psi::Ket)
    Lp = (b1.pmax - b1.pmin)
    dx = particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dx - Lp)/Lp > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(1im*b2.xmin*(particle.samplepoints(b1)-b1.pmin)).*psi.data
    psi_fft = exp(1im*b1.pmin*particle.samplepoints(b2)).*ifft(psi_shifted)*sqrt(N)
    return Ket(b2, psi_fft)
end

function transformation(b1::PositionBasis, b2::MomentumBasis, psi::Ket)
    Lx = (b1.xmax - b1.xmin)
    dp = particle.spacing(b2)
    if b1.N != b2.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    N = b1.N
    psi_shifted = exp(-1im*b2.pmin*(particle.samplepoints(b1)-b1.xmin)).*psi.data
    psi_fft = exp(-1im*b1.xmin*particle.samplepoints(b2)).*fft(psi_shifted)/sqrt(N)
    return Ket(b2, psi_fft)
end

psix0_fft = transformation(basis_position, basis_momentum, psix0)
psip0_fft = transformation(basis_momentum, basis_position, psip0)

@test 0.1 > abs(x0 - expect(opp_x, psix0_fft))
@test p0 ≈ expect(opp_p, psix0_fft)
@test 0.1 > abs(p0 - expect(opx_p, psip0_fft))
@test x0 ≈ expect(opx_x, psip0_fft)

@test 1e-12 > norm(psix0_fft - psip0)
@test 1e-12 > norm(psip0_fft - psix0)


Tpx = particle.FFTOperator(basis_momentum, basis_position)
Txp = particle.FFTOperator(basis_position, basis_momentum)
psix0_fft = Tpx*psix0
psip0_fft = Txp*psip0

@test 0.1 > abs(x0 - expect(opp_x, psix0_fft))
@test p0 ≈ expect(opp_p, psix0_fft)
@test 0.1 > abs(p0 - expect(opx_p, psip0_fft))
@test x0 ≈ expect(opx_x, psip0_fft)

@test 1e-12 > norm(psix0_fft - psip0)
@test 1e-12 > norm(psip0_fft - psix0)

@test 1e-12 > norm(dagger(Tpx*psix0) - dagger(psix0)*dagger(Tpx))
@test 1e-12 > norm(dagger(Txp*psip0) - dagger(psip0)*dagger(Txp))

psi_ = deepcopy(psip0)
operators.gemv!(Complex(1.), Tpx, psix0, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psip0)
operators.gemv!(Complex(1.), Tpx, psix0, Complex(1.), psi_)
@test 1e-12 > norm(psi_ - 2*psip0)

psi_ = deepcopy(psix0)
operators.gemv!(Complex(1.), Txp, psip0, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psix0)
operators.gemv!(Complex(1.), Txp, psip0, Complex(1.), psi_)
@test 1e-12 > norm(psi_ - 2*psix0)

rhox0x0 = tensor(psix0, dagger(psix0))
rhox0p0 = tensor(psix0, dagger(psip0))
rhop0x0 = tensor(psip0, dagger(psix0))
rhop0p0 = tensor(psip0, dagger(psip0))

rho_ = DenseOperator(basis_momentum, basis_position)
operators.gemm!(Complex(1.), Tpx, rhox0x0, Complex(0.), rho_)
@test 1e-12 > tracedistance(rho_, rhop0x0)
@test 1e-12 > tracedistance(Tpx*rhox0x0, rhop0x0)

rho_ = DenseOperator(basis_position, basis_momentum)
operators.gemm!(Complex(1.), rhox0x0, Txp, Complex(0.), rho_)
@test 1e-12 > tracedistance(rho_, rhox0p0)
@test 1e-12 > tracedistance(rhox0x0*Txp, rhox0p0)

rho_ = DenseOperator(basis_momentum, basis_momentum)
operators.gemm!(Complex(1.), Tpx, rhox0p0, Complex(0.), rho_)
@test 1e-12 > tracedistance(rho_, rhop0p0)
@test 1e-12 > tracedistance(Tpx*rhox0x0*Txp, rhop0p0)

rho_ = DenseOperator(basis_momentum, basis_momentum)
operators.gemm!(Complex(1.), rhop0x0, Txp, Complex(0.), rho_)
@test 1e-12 > tracedistance(rho_, rhop0p0)
@test 1e-12 > tracedistance(Txp*rhop0p0*Tpx, rhox0x0)

# Test FFT with lazy product
psi_ = deepcopy(psix0)
operators.gemv!(Complex(1.), LazyProduct(Txp, Tpx), psix0, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psix0)
@test 1e-12 > norm(Txp*(Tpx*psix0) - psix0)

psi_ = deepcopy(psix0)
I = full(identityoperator(basis_momentum))
operators.gemv!(Complex(1.), LazyProduct(Txp, I, Tpx), psix0, Complex(0.), psi_)
@test 1e-12 > norm(psi_ - psix0)
@test 1e-12 > norm(Txp*I*(Tpx*psix0) - psix0)

# Test dense FFT operator
Txp_dense = DenseOperator(Txp)
Tpx_dense = DenseOperator(Tpx)
@test typeof(Txp_dense) == DenseOperator
@test typeof(Tpx_dense) == DenseOperator
@test 1e-5 > tracedistance(Txp_dense*rhop0p0*Tpx_dense, rhox0x0)

end # testset
