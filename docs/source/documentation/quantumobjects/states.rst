.. _section-states:

States
^^^^^^

State vectors in **QuantumOptics.jl** are interpreted as coefficients in respect to a certain :ref:`basis <section-bases>`. For example the state :math:`|\psi\rangle` can be represented in the basis :math:`\{|u_i\rangle\}_i` as :math:`\psi_i`. These quantities are connected by

.. math::

    |\psi\rangle = \sum_i \psi_i |u_i\rangle

and the conjugate equation

.. math::

    \langle\psi| = \sum_i \psi_i^* \langle u_i|.

The distinction between coefficients in respect to bra or ket states is strictly enforced which guarantees that algebraic mistakes raise an explicit error::

    basis = FockBasis(2)
    x = Ket(basis, [1,1,1]) # Not necessarily normalized
    y = Bra(basis, [0,1,0])
    x + y # Error!

Many of the commonly used states are already implemented in the specific quantum systems, e.g. :jl:func:`spinup` for spins or :jl:func:`coherentstate` for systems described by a fock basis. The :jl:func:`basisstate` function is defined for every basis and is used to generate the i-th basis-state of this basis.

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

Alternatively, one can use the tensor function to create a density operator by combining a ket with a bra::

    tensor(x, dagger(x))

The following functions are also available for states:

    :jl:func:`norm`
    :jl:func:`normalize`
    :jl:func:`normalize!`
    :jl:func:`ptrace`
    :jl:func:`permutesystems`
