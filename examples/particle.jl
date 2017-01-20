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
basis_momentum = quantumoptics.particle.MomentumBasis(basis_position)
xsamplepoints = quantumoptics.particle.samplepoints(basis_position)
psamplepoints = quantumoptics.particle.samplepoints(basis_momentum)

Tpx = quantumoptics.particle.FFTOperator(basis_momentum, basis_position)
Txp = quantumoptics.particle.FFTOperator(basis_position, basis_momentum)

# Position Basis
#p_pos = SparseOperator(quantumoptics.particle.momentumoperator(basis_position))
p_pos = Txp*quantumoptics.particle.momentumoperator(basis_momentum)*Tpx
x_pos = quantumoptics.particle.positionoperator(basis_position)
V_pos = Operator(basis_position, diagm(omega^2*xsamplepoints.^2))
H_pos = p_pos*p_pos/2 + V_pos

psi0_pos = quantumoptics.particle.gaussianstate(basis_position, x0, p0, sigma0)
@time tout, psi_pos_t = timeevolution.schroedinger(T, psi0_pos, H_pos)
@time tout, psi_pos_t = timeevolution.schroedinger(T, psi0_pos, H_pos)


# Momentum Basis
p_mom = SparseOperator(quantumoptics.particle.momentumoperator(basis_momentum))
x_mom = SparseOperator(quantumoptics.particle.positionoperator(basis_momentum))
#x_mom = SparseOperator(Tpx*quantumoptics.particle.positionoperator(basis_position)*Txp)
V_mom = SparseOperator(Tpx*Operator(basis_position, diagm(omega^2*xsamplepoints.^2))*Txp)
H_mom = p_mom*p_mom/2 + V_mom

psi0_mom = quantumoptics.particle.gaussianstate(basis_momentum, x0, p0, sigma0)
@time tout, psi_mom_t = timeevolution.schroedinger(T, psi0_mom, H_mom)
@time tout, psi_mom_t = timeevolution.schroedinger(T, psi0_mom, H_mom)


# Mixed evolution
H_mix = LazyProduct(Txp, p_mom*p_mom/2, Tpx) + SparseOperator(V_pos)
# println(expect(H_mix, psi0_pos))
# println(expect(H_pos, psi0_pos))
# println(expect(H_mix, psi0_pos + 0.1im*H_mix*psi0_pos))
# println(expect(H_pos, psi0_pos + 0.1im*H_pos*psi0_pos))
@time tout, psi_mix_t = timeevolution.schroedinger(T, psi0_pos, H_mix)
@time tout, psi_mix_t = timeevolution.schroedinger(T, psi0_pos, H_mix)

for i=1:length(T)
    println(expect(p_mom, psi_mom_t[i]), ", ", expect(p_pos, psi_pos_t[i]))
    println(expect(x_mom, psi_mom_t[i]), ", ", expect(x_pos, psi_pos_t[i]))
    println(norm(Txp*psi_mom_t[i] - psi_pos_t[i]))
    println(norm(psi_mix_t[i] - psi_pos_t[i]))
    println("================================================")
end
