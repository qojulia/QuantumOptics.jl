
using QuantumOptics
using PyPlot

xmin = -3
xmax = 3
Nsteps = 200
L = xmax - xmin
m = 0.5
E0 = 20

b_position = PositionBasis(xmin, xmax, Nsteps)
xpoints = samplepoints(b_position)

x = position(b_position)
p = momentum(b_position)

potential = x -> E0 + x^4 - 8*x^2
V = potentialoperator(b_position, potential)
Hkin = p^2/2m
H = Hkin + full(V);

E, states = eig((H + dagger(H))/2, 1:4);
println(E)

figure(figsize=(5, 3))
subplot(2, 2, 1)
plot(xpoints, abs2(states[1].data))
subplot(2, 2, 2)
plot(xpoints, abs2(states[2].data))
subplot(2, 2, 3)
plot(xpoints, abs2(states[3].data))
subplot(2, 2, 4)
plot(xpoints, abs2(states[4].data))
tight_layout()

localizedstates = Ket[(states[1] - states[2])/sqrt(2),
                      (states[1] + states[2])/sqrt(2),
                      (states[3] - states[4])/sqrt(2),
                      (states[3] + states[4])/sqrt(2)]

figure(figsize=(10, 3))
subplot(1, 2, 1)
plot(xpoints, potential.(xpoints))
for state in localizedstates
    E = abs(expect(H, state))
    plot(xpoints[[1,end]], [E, E], "C1")
end

subplot(2, 4, 7)
plot(xpoints, abs2(localizedstates[1].data))
subplot(2, 4, 8)
plot(xpoints, abs2(localizedstates[2].data))
subplot(2, 4, 3)
plot(xpoints, abs2(localizedstates[3].data))
subplot(2, 4, 4)
plot(xpoints, abs2(localizedstates[4].data));
tight_layout()

b_sub = SubspaceBasis(b_position, localizedstates)
println("dim(subspace): ", length(b_sub))

P = projector(b_sub, b_position)
Pt = dagger(P)

x_sub = P*x*Pt
Hkin_sub = P*Hkin*Pt
V_sub = P*V*Pt
H_sub = P*H*Pt;
println("dim(H_sub): ", length(H_sub.basis_l), "x", length(H_sub.basis_r))

T = [0:0.1:23;]
psi0_sub = basisstate(b_sub, 1)

tout, psi_t = timeevolution.schroedinger(T, psi0_sub, H_sub)

figure(figsize=[10, 3])
subplot(1, 2, 1)
plot(tout, [abs2(psi.data[1]) for psi in psi_t])
plot(tout, [abs2(psi.data[2]) for psi in psi_t])

subplot(1, 2, 2)
# Plot particle probablity distribution in the super Hilbert space.
for i=1:10:length(T)
    plot(xpoints, abs2((Pt*psi_t[i]).data), "k", alpha=0.6*(T[i]/T[end])^4+0.2)
end
plot(xpoints, abs2((Pt*psi_t[1]).data), "C0", label="Initial state")
plot(xpoints, abs2((Pt*psi_t[end]).data), "C1", label="Final state")
legend();

states = fermionstates(b_sub, 2)
for i in 1:length(states)
    println("$i: ", states[i])
end

Nparticles = 2
b_mb = ManyBodyBasis(b_sub, bosonstates(b_sub, Nparticles));

Hkin_mb = manybodyoperator(b_mb, Hkin_sub)
V_mb = manybodyoperator(b_mb, V_sub)
H_mb = manybodyoperator(b_mb, H_sub);

n1_sub = basisstate(b_sub, 1) ⊗ dagger(basisstate(b_sub, 1))
n1_mb = manybodyoperator(b_mb, sparse(n1_sub));

n1_mb = number(b_mb, 1)
n2_mb = number(b_mb, 2)
n3_mb = number(b_mb, 3)
n4_mb = number(b_mb, 4);

psi0_mb = basisstate(b_mb, [2, 0, 0, 0])
tout, psi_t_mb = timeevolution.schroedinger(T, psi0_mb, H_mb);

"""
Sparse operator |x_i><x_i| in position basis.
"""
function nx(b::PositionBasis, i)
    op = SparseOperator(b)
    op.data[i, i] = 1.
    op
end

"""
Probability density in the position basis of the given many body state.
"""
function probabilitydensity_x(state)
    n = Vector{Float64}(length(b_position))
    for i=1:length(b_position)
        nx_i = nx(b_position, i)
        nx_i_sub = P*nx_i*Pt
        nx_i_mb = manybodyoperator(b_mb, nx_i_sub)
        n[i] = real(expect(nx_i_mb, state))
    end
    n
end;

figure(figsize=[10, 3])
subplot(1, 2, 1)
plot(tout, real(expect(n1_mb, psi_t_mb)))
plot(tout, real(expect(n2_mb, psi_t_mb)))

subplot(1, 2, 2)
T_ = tout[tout.<23]
plot(xpoints, real(probabilitydensity_x(psi_t_mb[1])), label="Initial state")
plot(xpoints, real(probabilitydensity_x(psi_t_mb[length(T_)])), label="Final state")
legend();

b2_position = b_position ⊗ b_position
x1 = embed(b2_position, 1, x)
x2 = embed(b2_position, 2, x);
r = x1 - x2
d = abs(diag(r.data)).^-1
d[d.==Inf] = d[2]
data = spdiagm(d)
H_coulomb = SparseOperator(b2_position, b2_position, data);

H_coulomb_sub = (P⊗P)*H_coulomb*(Pt⊗Pt);

H_coulomb_mb = manybodyoperator(b_mb, H_coulomb_sub);

E_coulomb_attractive, states_coulomb_attractive = eig(H_mb - H_coulomb_mb)
E_coulomb_repulsive, states_coulomb_repulsive = eig(H_mb + H_coulomb_mb);

"""
Probability density in the position basis of the given many body state.
"""
function probabilitydensity_x1x2(state, indices)
    P2 = P ⊗ P
    P2t = dagger(P2)
    n = Matrix{Float64}(length(indices), length(indices))
    for i=1:length(indices)
        nx_i_sub = P*nx(b_position, indices[i])*Pt
        for j=1:length(indices)
            nx_j_sub = P*nx(b_position, indices[j])*Pt
            nx_ij_sub = nx_i_sub ⊗ nx_j_sub
            nx_ij_mb = manybodyoperator(b_mb, nx_ij_sub)
            n[i, j] = real(expect(nx_ij_mb, state))
        end
    end
    n
end;

println(abs(E_coulomb_attractive)[1:4])

indices = [1:20:length(b_position);]

n_attractive1 = probabilitydensity_x1x2(states_coulomb_attractive[1] + states_coulomb_attractive[2], indices)
n_attractive2 = probabilitydensity_x1x2(states_coulomb_attractive[1] - states_coulomb_attractive[2], indices)
n_attractive3 = probabilitydensity_x1x2(states_coulomb_attractive[3] + states_coulomb_attractive[4], indices)
n_attractive4 = probabilitydensity_x1x2(states_coulomb_attractive[3] - states_coulomb_attractive[4], indices);

figure(figsize=[10,2])
subplot(1, 4, 1)
pcolor(n_attractive1)
subplot(1, 4, 2)
pcolor(n_attractive2)
subplot(1, 4, 3)
pcolor(n_attractive3)
subplot(1, 4, 4)
pcolor(n_attractive4);

println(abs(E_coulomb_repulsive)[1:5])

indices = [1:20:length(b_position);]

n_repulsive1 = probabilitydensity_x1x2(states_coulomb_repulsive[1], indices)
n_repulsive2 = probabilitydensity_x1x2(states_coulomb_repulsive[2], indices)
n_repulsive3 = probabilitydensity_x1x2(states_coulomb_repulsive[3], indices);

figure(figsize=[10,3])
subplot(1, 3, 1)
pcolor(n_repulsive1)
subplot(1, 3, 2)
pcolor(n_repulsive2)
subplot(1, 3, 3)
pcolor(n_repulsive3)
show();
