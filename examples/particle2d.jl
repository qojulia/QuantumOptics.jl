using QuantumOptics

N = 200
xmin = -10.
xmax = 10.
omega = 2.

x0 = 1.
p0 = 0.
sigma0 = 1.

T = [0.:0.2:2.;]


basis_position = quantumoptics.particle.PositionBasis(xmin, xmax, N)
basis_position2d = compose(basis_position, basis_position)
basis_momentum = quantumoptics.particle.MomentumBasis(basis_position)
basis_momentum2d = compose(basis_momentum, basis_momentum)

xsamplepoints = quantumoptics.particle.samplepoints(basis_position)
psamplepoints = quantumoptics.particle.samplepoints(basis_momentum)

# Tpx = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)
# Txp = quantumoptics.particle.FFTOperator(basis_position, basis_momentum)

# Position Basis
@time p = SparseOperator(quantumoptics.particle.momentumoperator(basis_position))
@time x = SparseOperator(quantumoptics.particle.positionoperator(basis_position))
@time V = SparseOperator(basis_position, diagm(complex(omega)^2*xsamplepoints.^2))
@time I = identityoperator(basis_position)

@time p2 = p^2/2
@time Hkin2d = (tensor(p2, I) + tensor(I, p2))
@time V2d = tensor(V, I) + tensor(I, V)
@time H2d = Hkin2d + V2d

@time psi0 = quantumoptics.particle.gaussianstate(basis_position, x0, p0, sigma0)
@time psi0_2d = tensor(psi0, psi0)
@time tout, psi_t = timeevolution.schroedinger(T, psi0_2d, H2d)
@time tout, psi_t = timeevolution.schroedinger(T, psi0_2d, H2d)


# for i=1:length(T)
#     println(expect(p_mom, psi_mom_t[i]), ", ", expect(p_pos, psi_pos_t[i]))
#     println(expect(x_mom, psi_mom_t[i]), ", ", expect(x_pos, psi_pos_t[i]))
#     println(norm(Txp*psi_mom_t[i] - psi_pos_t[i]))
#     println(norm(psi_mix_t[i] - psi_pos_t[i]))
#     println("================================================")
# end
