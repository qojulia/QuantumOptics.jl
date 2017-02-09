.. _section-statesandoperators:

States and Operators
====================

Here we will give a brief overview over the operations that are available for states and operators. All states and operators are defined on a basis, such as for example the :jl:func:`FockBasis`. For a full list of implemented bases and their predefined states and operators please refer to :ref:`Quantumsystems <section-quantumsystems>`. More details on the different types of operators can be found in :ref:`Operators <section-operators-detail>`.

.. _section-states:

States
^^^^^^

State vectors in **QuantumOptics.jl** are interpreted as coefficients in respect to a certain :ref:`basis <section-bases>`. For example the particle state :math:`|\Psi\rangle` can be represented in a (discrete) real space basis :math:`\{|x_i\rangle\}_i` as :math:`\Psi(x_i)`. These quantities are connected by

.. math::

    |\Psi\rangle = \sum_i \Psi(x) |x_i\rangle

and the conjugate equation

.. math::

    \langle\Psi| = \sum_i \Psi(x)^* \langle x_i|

The distinction between coefficients in respect to bra or ket states is strictly enforced which guarantees that algebraic mistakes raise an explicit error::

    basis = FockBasis(3)
    x = Ket(basis, [1,1,1]) # Not necessarily normalized
    y = Bra(basis, [0,1,0])

Many commonly used states are already implemented for various systems, like e.g. :jl:func:`fockstate(n::Int)` or :jl:func:`gaussianstate(::MomentumBasis, x0, p0, sigma)`.

All expected arithmetic functions like \*, /, +, - are implemented::

    x + x
    x - x
    2*x
    y*x # Inner product

The hermitian conjugate is performed by the :jl:func:`dagger(x::Ket)` function which transforms a bra in a ket and vice versa::

    dagger(x) # Bra(basis, [1,1,1])

Composite states can be created with the :jl:func:`tensor(x::Ket, y::Ket)` function or with the equivalent :math:`\otimes` operator::

    tensor(x, x)
    x ⊗ x
    tensor(x, x, x)

The following functions are also available for states:

* Normalization functions:
    :jl:func:`norm(x::Ket, p=2)`
    :jl:func:`normalize(x::Ket, p=2)`
    :jl:func:`normalize!(x::Ket, p=2)`

* Partial trace
    :jl:func:`ptrace(x::Ket, indices::Vector{Int})`
    :jl:func:`ptrace(x::Bra, indices::Vector{Int})`


.. _section-operators:

Operators
^^^^^^^^^

Operators can be defined as linear mappings from one Hilbert space to another. However, equivalently to states, operators in **QuantumOptics.jl** are interpreted as coefficients of an abstract operator in respect to one or more generally two, possibly distinct :ref:`bases <section-bases>`. For a certain choice of bases :math:`\{|u_i\rangle\}_i` and :math:`\{|v_j\rangle\}_j` an abstract operator :math:`A` has the coefficients :math:`A_{ij}` which are connected by the relation

.. math::

    A =  \sum_{ij} A_{ij} | u_i \rangle \langle v_j |

All standard arithmetic functions for operators are defined, \*, /, +, -::

    b = SpinBasis(1//2)
    sx = sigmax(b)
    sy = sigmay(b)
    sx + sy
    sx * sy # Matrix product
    sx ⊗ sy

Additionally the following functions are implemented (for :jl:func:`A::Operator`, :jl:func:`B::Operator`):

* Hermitian conjugate:
    :jl:func:`dagger(A)`

* Normalization:
    :jl:func:`trace(A)`
    :jl:func:`norm(A)`
    :jl:func:`normalize(A)`
    :jl:func:`normalize!(A)`

* Expectation values:
    :jl:func:`expect(A, B)`


* Tensor product:
    :jl:func:`tensor(A, B)`

* Partial trace:
    :jl:func:`ptrace(A, index::Int)`
    :jl:func:`ptrace(A, indices::Vector{Int})`

* Creating operators from states:
    :jl:func:`tensor(x::Ket, y::Bra)`
    :jl:func:`projector(x::Ket, y::Bra)`

For creating operators of the type :math:`A = I \otimes I \otimes ... a_i ... \otimes I` the very useful embed function can be used:

* :jl:func:`embed(b::Basis, index::Int, op::Operator)`
* :jl:func:`embed(b::Basis, indices::Vector{Int}, ops::Vector{T <: Operator})`

E.g. for a system consisting of 3 spins one can define the basis with::

    b_spin = SpinBasis(1//2)
    b = b_spin ⊗ b_spin ⊗ b_spin

An operator in this basis b that only acts on the second spin could be created as::

    identityoperator(b_spin) ⊗ sigmap(b_spin) ⊗ identityoperator(b_spin)

Equivalently, the embed function simplifies this to::

    embed(b, 2, sigmap(b_spin))
