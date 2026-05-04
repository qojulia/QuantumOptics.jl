# Monte Carlo wave function time evolution

Instead of solving the Master equation

```math
\dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
             + \sum_i \big(
                    J_i \rho J_i^\dagger
                    - \frac{1}{2} J_i^\dagger J_i \rho
                    - \frac{1}{2} \rho J_i^\dagger J_i
                \big)
```

directly, one can use the quantum jump formalism to evaluate single stochastic quantum trajectories using the Monte Carlo wave function method. For large numbers of trajectories the statistical average then approximates the result of the Master equation. The huge advantage is that instead of describing the state of the quantum system by a density matrix of size ``N^2`` these trajectories work in terms of state vectors of size ``N``. This is somewhat negated by the stochastic nature of the formalism which makes it necessary to repeat the simulation until the wanted accuracy is reached. It turns out, however, that for many cases, especially for high dimensional quantum systems, the necessary number of repetitions is much smaller than the system size ``N`` and therefore using the MCWF method is advantageous.

Additionally this quantum jump formalism also has a very intuitive physical interpretation. It basically describes the situation where every quantum jump, e.g. the emission of a photon, is detected by a detector and therefore the time evolution can be completely reconstructed by an outside observer. Depending on the efficiency of the used detectors this might be a much better description for an actual experiment.

This physical picture can be used to easily understand the actual MCWF algorithm

1.- Calculate coherent time evolution according to a Schroedinger equation with non-hermitian Hamiltonian ``H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i``

```math
i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H_\mathrm{nh} |\Psi(t)\rangle
```

2.- Since the Hamiltonian is non-hermitian the norm of the quantum state is not conserved and actually decreases with time. This can be interpreted in the way that the smaller the norm of the state gets the more probable it is that a quantum jump occurs. Quantitatively this means that the coherent time evolution stops when ``\langle \Psi(t)|\Psi(t)\rangle < p`` where ``p`` is a randomly generated number between 0 and 1.

3.- At these randomly determined times a quantum jump according to

```math
|\Psi(t)\rangle \rightarrow \frac{J_i |\Psi(t)\rangle}{||J_i |\Psi(t)\rangle||}
```

  is performed.

4.- Continue with coherent time evolution.

The stochastic average of these trajectories is then equal to the solution of the master equation ``\rho(t)``

```math
\lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N |\Psi^k(t)\rangle\langle\Psi^k(t)| = \rho(t)
```

and also the stochastic average of the single trajectory expectation values is equal to the expectation value according to the master equation


```math
\lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N \langle\Psi^k(t)| A |\Psi^k(t)\rangle = \mathrm{Tr}\big\{A \rho(t)\big\}
```

avoiding explicit calculations of density matrices.

The function computing a time evolution with the MCWF method can be called analogously to [`timeevolution.master`](@ref), namely with:

```@example mcwf
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
a = destroy(b) # hide
J = [a] # hide
tspan = [0,0.1] # hide
psi0 = fockstate(b, 0) # hide
tout, psit = timeevolution.mcwf(tspan, psi0, H, J)
nothing # hide
```

Since this function only calculates state vectors (as explained above), it requires the initial state in the form of a [`Ket`](@ref). Without any additional keyword arguments the [`timeevolution.mcwf`](@ref) function returns two vectors, the first contains the times identical to tspan and the second the state of the system at these points of time. However, finer control of the output is possible by using the keyword arguments `display_beforeevent` and `display_afterevent`. If they are set to `true` the state of the system immediately before and after a jump is added to these vectors. To avoid keeping all the states in memory when only certain expectation values are needed, one can optionally pass a function `fout(t, psi)` which is called at every specified point of time:

```@example mcwf
function fout(t, psi)
    expect(a, psi)/norm(psi)^2
end
tout, exp_a = timeevolution.mcwf(tspan, psi0, H, J; fout=fout)
nothing # hide
```

Note, that the given state is not normalized and is still in use by the ode solver. Therefore it should not be changed and if one wants to store the state at a certain point of time it has to be copied explicitly.

As shown above, the concept of the MCWF method requires jumps of the form ``J_i|\psi\rangle`` to be applied at certain times. Therefore, it is not possible to calculate the time evolution according to a MCWF for Lindblad terms of the form ``\sum_{i,j}\gamma_{ij}\left(J_i\rho J_j^\dagger - J_i^\dagger J_j\rho/2 - \rho J_i^\dagger J_j/2\right)``, since such a Lindblad term relies on the the multiplication of two different jump operators ``J_i`` and ``J_j`` from the left and right, respectively, with the density matrix ``\rho``. However, it is possible to write such a Lindblad term in diagonal form ``\sum_i d_i \left(D_i\rho D_i^\dagger - D_i^\dagger D_i\rho - \rho D_i^\dagger D_i\right)``. Here, ``d_i`` are the eigenvalues of the matrix with entries ``\gamma_{ij}`` and the diagonal jump operators ``D_i`` are defined by

```math
D_i = \sum_k v^{(i)}_k J_k,
```

where ``v^{(i)}`` is the eigenvector corresponding to the eigenvalue ``d_i``.

This diagonalization is implemented with the function

* [`diagonaljumps(rates::Array{Float64}, J::Vector)`](@ref)

which returns the eigenvalues (new decay rates) and the corresponding set of jump operators. As a quick example, say you have three spin-1/2 particles that decay collectively. When using the master equation solver you can simply pass the matrix containing the collective decay rates as a keyword argument. To do the same with the MCWF we need to do:

```@example mcwf
spinbasis = SpinBasis(1//2)
threespinbasis = spinbasis ⊗ spinbasis ⊗ spinbasis
sm(i) = embed(threespinbasis, i, sigmam(spinbasis))

a, b, c = 1.0, 0.5, 0.2
rates = [a b c; b a c; c b a]
J = [sm(i) for i=1:3]
d, D = diagonaljumps(rates, J)
nothing # hide
```

Now, we can call the solver with the acquired jump operators `D` multiplied by their corresponding rates `d`:

```@example mcwf
H = SparseOperator(threespinbasis) # hide
psi0 = basisstate(threespinbasis, 1) # hide
tout, psi_t = timeevolution.mcwf(tspan, psi0, H, D; rates=d)
nothing # hide
```

## [Functions](@id mcwf: Functions)

* [`timeevolution.mcwf`](@ref)
* [`timeevolution.mcwf_h`](@ref)
* [`timeevolution.mcwf_nh`](@ref)
* [`timeevolution.mcwf_dynamic`](@ref)
* [`diagonaljumps`](@ref)

## [Examples](@id mcwf: Examples)

* [Pumped cavity](@ref)
* [Jaynes-Cummings model](@ref)
