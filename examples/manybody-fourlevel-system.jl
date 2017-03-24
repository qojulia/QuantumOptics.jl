
using QuantumOptics
using PyPlot

b = NLevelBasis(4)

H = diagonaloperator(b, [0, 1, 2, 3]);

j34 = transition(b, 3, 4) # decay from 4th into 3rd level
j23 = transition(b, 2, 3) # decay from 3rd into 2nd level
j12 = transition(b, 1, 2) # decay from 2nd into 1st level;

b_mb = ManyBodyBasis(b, fermionstates(b, 2))

H_mb = manybodyoperator(b_mb, H)

j34_mb = manybodyoperator(b_mb, j34)
j23_mb = manybodyoperator(b_mb, j23)
j12_mb = manybodyoperator(b_mb, j12)

Γ = [3., 1., 0.5]
J_mb = [j34_mb, j23_mb, j12_mb];

T = [0:0.1:10;]
Ψ0_mb = basisstate(b_mb, [0, 0, 1, 1])
tout, ρt = timeevolution.master(T, Ψ0_mb, H_mb, J_mb; Gamma=Γ);

psi0011 = basisstate(b_mb, [0, 0, 1, 1])
psi1100 = basisstate(b_mb, [1, 1, 0, 0])

n0011 = psi0011 ⊗ dagger(psi0011)
n1100 = psi1100 ⊗ dagger(psi1100)

figure(figsize=(10, 3))
subplot(1, 2, 1)
plot(tout, real(expect(n0011, ρt)), label="n0011")
plot(tout, real(expect(n1100, ρt)), label="n1100")
legend()

subplot(1, 2, 2)
for i in 1:4
    plot(tout, real(expect(number(b_mb, i), ρt)), label="level $i")
end
legend();

b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2, 3]))

H_mb = manybodyoperator(b_mb, H)

j34_mb = manybodyoperator(b_mb, j34)
j23_mb = manybodyoperator(b_mb, j23)
j12_mb = manybodyoperator(b_mb, j12);

a1_mb = destroy(b_mb, 1) # Particles are lost from the first level

Γ = [0.9, 0.5, 0.3, 3.0]
J_mb = [j34_mb, j23_mb, j12_mb, a1_mb]

T = [0:0.1:10;]
Ψ0_mb = basisstate(b_mb, [0, 0, 0, 3]) # Initially, all particles are in the uppermost level
tout, ρt = timeevolution.master(T, Ψ0_mb, H_mb, J_mb; Gamma=Γ);

figure(figsize=(5, 3))
plot(tout, real(expect(number(b_mb), ρt)), "--k", alpha=0.6, label="particle number")
for i in 1:4
    plot(tout, real(expect(number(b_mb, i), ρt)), label="level $i")
end
legend()
show()


