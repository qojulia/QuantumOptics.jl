
using QuantumOptics
using PyPlot

# Parameters
N_cutoff = 10

ωc = 0.1
ωa = 0.1
Ω = 1.;

# Bases
b_fock = FockBasis(N_cutoff)
b_spin = SpinBasis(1//2)
b = b_fock ⊗ b_spin;

# Fundamental operators
a = destroy(b_fock)
at = create(b_fock)
n = number(b_fock)

sm = sigmam(b_spin)
sp = sigmap(b_spin)
sz = sigmaz(b_spin)

# Hamiltonian
Hatom = ωa*sz/2
Hfield = ωc*n
Hint = Ω*(at⊗sm + a⊗sp)
H = identityoperator(b_fock)⊗Hatom + Hfield⊗identityoperator(b_spin) + Hint;

# Initial state
α = 1.
Ψ0 = coherentstate(b_fock, α) ⊗ spindown(b_spin)

# Integration time
T = [0:0.1:20;]

# Schroedinger time evolution
tout, Ψt = timeevolution.schroedinger(T, Ψ0, H);

exp_n = real(expect(n ⊗ identityoperator(b_spin), Ψt))
exp_sz = real(expect(identityoperator(b_fock) ⊗ sz, Ψt));

figure(figsize=(9,3))
subplot(1,2,1)
ylim([0, 2])
plot(T, exp_n);
xlabel(L"T")
ylabel(L"\langle n \rangle")

subplot(1,2,2)
ylim([-1, 1])
plot(T, exp_sz);
xlabel(L"T")
ylabel(L"\langle \sigma_z \rangle")

tight_layout();

γ = 0.5
J = [sqrt(γ)*identityoperator(b_fock) ⊗ sm];

# Master
tout, ρt = timeevolution.master(T, Ψ0, H, J)
exp_n_master = real(expect(n ⊗ identityoperator(b_spin), ρt))
exp_sz_master = real(expect(identityoperator(b_fock) ⊗ sz, ρt))

figure(figsize=(9,3))
subplot(1,2,1)
ylim([0, 2])
plot(T, exp_n_master);
xlabel(L"T")
ylabel(L"\langle n \rangle")

subplot(1,2,2)
ylim([-1, 1])
plot(T, exp_sz_master);
xlabel(L"T")
ylabel(L"\langle \sigma_z \rangle");

tight_layout();

# Monte Carlo wave function
tout, Ψt = timeevolution.mcwf(T, Ψ0, H, J; seed=2,
                                display_beforeevent=true,
                                display_afterevent=true)
exp_n_mcwf = real(expect(n ⊗ identityoperator(b_spin), Ψt))
exp_sz_mcwf = real(expect(identityoperator(b_fock) ⊗ sz, Ψt))

figure(figsize=(9,3))
subplot(1,2,1)
ylim([0, 2])
plot(tout, exp_n_mcwf)
xlabel(L"T")
ylabel(L"\langle n \rangle")

subplot(1,2,2)
ylim([-1, 1])
plot(tout, exp_sz_mcwf)
xlabel(L"T")
ylabel(L"\langle \sigma_z \rangle");

tight_layout();

Ntrajectories = 10
exp_n_average = zeros(Float64, length(T))
exp_sz_average = zeros(Float64, length(T))

for i = 1:Ntrajectories
    tout, Ψt = timeevolution.mcwf(T, Ψ0, H, J; seed=i)
    exp_n_average += real(expect(n ⊗ identityoperator(b_spin), Ψt))
    exp_sz_average += real(expect(identityoperator(b_fock) ⊗ sz, Ψt))
end

exp_n_average /= Ntrajectories
exp_sz_average /= Ntrajectories

figure(figsize=(9,3))
subplot(1,2,1)
ylim([0, 2])
plot(T, exp_n_master)
plot(T, exp_n_average)
xlabel(L"T")
ylabel(L"\langle n \rangle")

subplot(1,2,2)
ylim([-1, 1])
plot(T, exp_sz_master)
plot(T, exp_sz_average)
xlabel(L"T")
ylabel(L"\langle \sigma_z \rangle");

tight_layout();


