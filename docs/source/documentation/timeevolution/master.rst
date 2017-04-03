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

The arguments required are quite similar to the ones of :jl:func:`schroedinger`. ``tspan`` is a vector of times, ``rho0`` the initial state and ``H`` the Hamiltonian. We now also need the vector ``J`` that specifies the jump operators of the system.

The additional arguments available are

* ``Gamma::{Vector{Float64}, Matrix{Float64}}``
* ``Jdagger::Vector``
* ``fout::Function``

The first specifies the decay rates of the system with default values one. If ``Gamma`` is a vector of length `length(J)`, then the `i` th entry of ``Gamma`` is paired with the `i`-th entry of ``J``, such that :math:`J_i` decays with :math:`\gamma_i`. If, on the other hand, ``Gamma`` is a matrix, then all entries of ``J`` are paired with one another and matched with the corresponding entrie of ``Gamma``, resulting
in a Lindblad term of the form :math:`\sum_{i,j}\gamma_{ij}\left(J_i\rho J_j^\dagger - J_i^\dagger J_j\rho/2 - \rho J_i^\dagger J_j/2\right)`.

The second keyword argument can be used to pass a specific set of jump operators to be used in place of all :math:`J^\dagger` appearances in the Lindblad term.

We can pass an output function just like the one for a Schrödinger equation. Note, though, that now the function must be defined with the arguments ``fout(t, rho)``.

Furthermore, a time-dependent Hamiltonian can also be implemented analogously to a Schrödinger equation using :jl:func:`master_dynamic(tspan, rho0, f)`.

For performance reasons the solver internally first creates the non-hermitian Hamiltonian :math:`H_\mathrm{nh} = H - \frac{i\hbar}{2} \sum_i J_i^\dagger J_i` and solves the equation

.. math::

    \dot{\rho} = -\frac{i}{\hbar} \big[H_\mathrm{nh},\rho\big]
                 + \sum_i J_i \rho J_i^\dagger

If for any reason this behavior is unwanted, e.g. special operators are used that don't support addition, the function master_h (h for hermitian) can be used.

* :func:`master_h(tspan, rho0::DenseOperator, H::Operator, J::Vector)`

* :func:`master_nh(tspan, rho0::DenseOperator, Hnh::Operator, J::Vector)`


Examples
--------

* :ref:`example-pumped-cavity`
* :ref:`example-jaynes-cummings`
* :ref:`example-raman`
* :ref:`example-manybody-fourlevel-system`
