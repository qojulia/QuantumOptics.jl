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

* :jl:func:`schroedinger`


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

For performance reasons the solver internally first creates the non-hermitian Hamiltonian :math:`H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i` and solves the equation

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H_\mathrm{nh},\rho\big]
                 + \sum_i J_i \rho J_i^\dagger

If for any reason this behavior is unwanted, e.g. special operators are used that don't support addition, the function master_h (h for hermitian) can be used.

* :func:`master(tspan, rho0::DenseOperator, H::Operator, J::Vector)`

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

* :func:`master(tspan, psi0::Ket, H::Operator, J::Vector)`
