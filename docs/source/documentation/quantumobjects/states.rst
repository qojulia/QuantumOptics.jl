.. _section-states:

States
^^^^^^

State vectors in **QuantumOptics.jl** are interpreted as coefficients in respect to a certain :ref:`basis <section-bases>`. For example the particle state :math:`|\Psi\rangle` can be represented in a (discrete) real space basis :math:`\{|x_i\rangle\}_i` as :math:`\Psi(x_i)`. These quantities are connected by

.. math::

    |\Psi\rangle = \sum_i \Psi(x_i) |x_i\rangle

and the conjugate equation

.. math::

    \langle\Psi| = \sum_i \Psi(x_i)^* \langle x_i|

The distinction between coefficients in respect to bra or ket states is strictly enforced which guarantees that algebraic mistakes raise an explicit error::

    basis = FockBasis(2)
    x = Ket(basis, [1,1,1]) # Not necessarily normalized
    y = Bra(basis, [0,1,0])
    x + y # Error!

Many of the often used states are already implemented in the various specific quantum systems, e.g. :jl:func:`spinup` for spins. :jl:func:`basisstate` is defined for every basis.

All expected arithmetic functions like \*, /, +, - are implemented::

    x + x
    x - x
    2*x
    y*x # Inner product

The hermitian conjugate is performed by the :jl:func:`dagger` function which transforms a bra in a ket and vice versa::

    dagger(x) # Bra(basis, [1,1,1])

Composite states can be created with the :jl:func:`tensor` function or with the equivalent :math:`\otimes` operator::

    tensor(x, x)
    x ⊗ x
    tensor(x, x, x)

The following functions are also available for states:

* Normalization functions
    :jl:func:`norm`
    :jl:func:`normalize`
    :jl:func:`normalize!`

* Partial trace
    :jl:func:`ptrace`

* Permutation of subsystems
    :jl:func:`permutesystems`
