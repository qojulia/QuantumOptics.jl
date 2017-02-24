.. _section-timeevolution:

Time-evolution
==============

**QuantumOptics.jl** implements solver for dynamics of closed and open quantum systems:

* :ref:`Schroedinger equation<section-schroedinger>`
* :ref:`Master equation<section-master>`
* :ref:`Monte Carlo wave function method (MCWF)<section-mcwf>`

The interfaces are designed to be as consistent as possible to make it easy to switch between different methods.


.. _section-schroedinger:

Schroedinger time evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Schroedinger equation as one of the basic postulates of quantum mechanics describes the dynamics of a quantum state in a closed quantum system. In Dirac notation the Schroedinger equation and its adjoint equation read

.. math::

    i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H |\Psi(t)\rangle

    - i\hbar\frac{\mathrm{d}}{\mathrm{d} t} \langle \Psi(t)| = \langle\Psi(t)| H

Both versions are implemented and are chosen automatically depending on the type of the provided initial state (Bra or Ket):

* :func:`schroedinger(tspan::Vector{Float64}, psi0::{Ket,Bra}, H::Operator)`

The Schrödinger equation solver requires the arguments :func:`tspan`, which is a vector containing the times, the initial state :func:`psi0`
as :func:`Ket` or :func:`Bra` and the Hamiltonian :func:`H`.

Additionally, one can pass an output function :func:`fout` as keyword argument. This can be convenient if one directly wants to compute a value that depends on the states, e.g. an expectation value, instead
of the states themselves. Consider, for example, a time evolution according to a Schrödinger equation where for all times we want to compute the expectation value of the operator :func:`A`. We can do this by::

    tout, psi_t = timeevolution.schroedinger(T, psi0, H)
    exp_val = expect(A, psi_t)

or equivalently::

    tout = Float64[]
    exp_val = Complex128[]
    function exp(t, psi)
      push!(tout, t)
      push!(exp_val, expect(A, psi)
    end
    timeevolution.schroedinger(T, psi0, H; fout=exp)

Although the method using :func:`fout` might seem more complicated, it can be very useful for large systems to save memory since instead of all the states we only store one complex number per time step. Note, that
:func:`fout` must always be defined with the arguments :func:`(t, psi)`. If :func:`fout` is given, all variables are assigned within :func:`fout` and the call to :func:`schroedinger`
returns :func:`nothing`.

We can also calculate the time evolution for a Hamiltonian that is time-dependent. In that case, we need to use the function :func:`schroedinger_dynamic(tspan, psi0, f::Function)`. As you can see, this function
requires the same arguments as :func:`schroedinger`, but a function :func:`f` instead of a Hamiltonian. As a brief example, consider a spin-1/2 particle that is coherently driven by a laser that has an amplitude that
varies in time. We can implement this with::

  basis = SpinBasis(1//2)
  ψ₀ = spindown(basis)
  function pump(t, psi)
    return sin(t)*(sigmap(basis) + sigmam(basis))
  end
  tspan = [0:0.1:10;]
  tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, pump)

.. _section-master:

Master time evolution
^^^^^^^^^^^^^^^^^^^^^

The dynamics of open quantum systems are governed by a master equation in Lindblad form:

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
                 + \sum_i \big(
                        J_i \rho J_i^\dagger
                        - \frac{1}{2} J_i^\dagger J_i \rho
                        - \frac{1}{2} \rho J_i^\dagger J_i
                    \big)

It is implemented by the function

* :func:`master(tspan, rho0::DenseOperator, H::Operator, J::Vector)`

The arguments required are quite similar to the ones of :func:`schroedinger`. :func:`tspan` is a vector of times, :func:`rho0` the initial state and :func:`H` the Hamiltonian. We now also need the vector :func:`J`
that specifies the jump operators of the system.

The additional arguments available are

* :func:`Gamma::{Vector{Float64}, Matrix{Float64}}`
* :func:`Jdagger::Vector`
* :func:`fout::Function`

The first specifies the decay rates of the system with default values one. If :func:`Gamma` is a vector of length :func:`length(J)`, then the `i` th entry of :func:`Gamma` is paired with the `i` th entry of :func:`J`, such
that :math:`J_i` decays with :math:`\gamma_i`. If, on the other hand, :func:`Gamma` is a matrix, then all entries of :func:`J` are paired with one another and matched with the corresponding entrie of :func:`Gamma`, resulting
in a Lindblad term of the form :math:`\sum_{i,j}\gamma_{ij}J_i\rho J_j^\dagger - J_i^\dagger J_j\rho/2 - \rho J_i^\dagger J_j/2`.

The second keyword argument can be used to pass a specific set of jump operators to be used in place of all :math:`J^\dagger` appearances in the Lindblad term.

We can pass an output function just like the one for a Schrödinger equation. Note, though, that now the function must be defined with the arguments :func:`fout(t, rho)`.

Furthermore, a time-dependent Hamiltonian can also be implemented analogously to a Schrödinger equation using :func:`master_dynamic(tspan, rho0, f)`.

For performance reasons the solver internally first creates the non-hermitian Hamiltonian :math:`H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i` and solves the equation

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H_\mathrm{nh},\rho\big]
                 + \sum_i J_i \rho J_i^\dagger

If for any reason this behavior is unwanted, e.g. special operators are used that don't support addition, the function master_h (h for hermitian) can be used.

* :func:`master_h(tspan, rho0::DenseOperator, H::Operator, J::Vector)`

* :func:`master_nh(tspan, rho0::DenseOperator, Hnh::Operator, J::Vector)`


.. _section-mcwf:

MCWF time evolution
^^^^^^^^^^^^^^^^^^^

Instead of solving the Master equation

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H,\rho\big]
                 + \sum_i \big(
                        J_i \rho J_i^\dagger
                        - \frac{1}{2} J_i^\dagger J_i \rho
                        - \frac{1}{2} \rho J_i^\dagger J_i
                    \big)

directly, one can use the quantum jump formalism to evaluate single stochastic quantum trajectories using the Monte Carlo wave function method. For large numbers of trajectories the statistical average then approximates the result of the Master equation. The huge advantage is that instead of describing the state of the quantum system by a density matrix of size :math:`N^2` these trajectories work in terms of state vectors of size :math:`N`. This is somewhat negated by the stochastic nature of the formalism which makes it necessary to repeat the simulation until the wanted accuracy is reached. It turns out, however, that for many cases, especially for high dimensional quantum systems, the necessary number of repetitions is much smaller than the system size :math:`N` and therefore using the MCWF method is advantageous.

Additionally this quantum jump formalism also has a very intuitive physical interpretation. It basically describes the situation where every quantum jump, e.g. the emission of a photon, is detected by a detector and therefore the time evolution can be completely reconstructed by an outside observer. Depending on the efficiency of the used detectors this might be a much better description for an actual experiment.

This physical picture can be used to easily understand the actual MCWF algorithm:

#. Calculate coherent time evolution according to a Schroedinger equation with non-hermitian Hamiltonian :math:`H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i`

    .. math::

        i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H_\mathrm{nh} |\Psi(t)\rangle

#. Since the Hamiltonian is non-hermitian the norm of the quantum state is not conserved and actually decreases with time. This can be interpreted in the way that the smaller the norm of the state gets the more probable it is that a quantum jump occurs. Quantitatively this means that the coherent time evolution stops when :math:`\langle \Psi(t)|\Psi(t)\rangle < p` where :math:`p` is a randomly generated number between 0 and 1.

#. At these randomly determined times a quantum jump according to

    .. math::

        |\Psi(t)\rangle \rightarrow \frac{J_i |\Psi(t)\rangle}{||J_i |\Psi(t)\rangle||}

    is performed.

#. Continue with coherent time evolution.

The stochastic average of these trajectories is then equal to the solution of the master equation :math:`\rho(t)`

.. math::

    \lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N |\Psi^k(t)\rangle\langle\Psi^k(t)| = \rho(t)

and also the stochastic average of the single trajectory expectation values is equal to the expectation value according to the master equation

.. math::

    \lim\limits_{N \rightarrow \infty}\frac{1}{N} \sum_{k=1}^N \langle\Psi^k(t)| A |\Psi^k(t)\rangle = \mathrm{Tr}\big\{A \rho(t)\big\}

avoiding explicit calculations of density matrices.

The function computing a time evolution with the MCWF method can be called analogously to :func:`master`, namely with

* :func:`mcwf(tspan, psi0::Ket, H::Operator, J::Vector)`

Since this function only calculates state vectors (as explained above), it requires the initial state in the form of a ket.


Advanced examples
^^^^^^^^^^^^^^^^^

This section is meant to provide a basic introduction to the implemented time evolution solvers and illustrate some simple examples.
Most applications of the toolbox involve the simulation of a time evolution in one way or another, so please refer to :ref:`section-examples`
for more sophisticated uses of the solvers.
