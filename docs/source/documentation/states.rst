.. _section-statesandoperators:

States and Operators
====================

States and operators are the fundamental building blocks used in **QuantumOptics.jl**, which makes it essential to get an intuitive understanding of the underlying concepts of their implementation. Physically, quantum states are abstract vectors in a Hilbert space and operators are linear functions that map these states from one Hilbert space to another. To perform actual numerical calculations with these objects, it is necessary to choose a basis of the Hilbert space under consideration and use their numerical representation relative to this basis. For example if the states :math:`\{|u_i\rangle\}_i` form a basis of a Hilbert space, every possible state :math:`|\psi\rangle` can be expressed as coordinates :math:`\psi_i` in respect to this basis:

.. math::

    |\psi\rangle = \sum_i |u_i\rangle\langle u_i |\psi \rangle
                 = \sum_i \psi_i |u_i\rangle

In **QuantumOptics.jl** all states therefor contain two types of information. The choice of basis and the coordinates of the state in respect to this basis.

Operators are implemented in a very similar fashion. The only thing that makes it more complicated is that in principle it is possible to choose different bases for the left and right hand side, which sometimes is quite useful. Assuming that the states :math:`\{|u_i\rangle\}_{i=1}^{N_u}` form a basis of one Hilbert space :math:`\mathcal{H}_u` and the states :math:`\{|v_i\rangle\}_{i=1}^{N_v}` form a basis of another Hilbert space :math:`\mathcal{H}_v`, every operator defined as map from :math:`\mathcal{H}_v` to :math:`\mathcal{H}_u` can be expressed as coordinates :math:`A_{ij}` in respect to these two bases:

.. math::

    \hat{A} = \sum_{ij} |u_i\rangle\langle u_i| \hat{A} |v_j\rangle\langle v_j|
            = \sum_{ij} A_{ij} |u_i\rangle \langle v_j|

Since for operators two different Hilbert spaces are involved every operator in **QuantumOptics.jl** has to store information about the left-hand basis, the right-hand basis as well as the coordinates of the operator in respect to these two bases.

The fact that **QuantumOptics.jl** knows about the choice of basis for every quantum object means that it can check if all performed operations physically make sense, catching many possible mistakes early on. Additionally, explicitly specifying a basis makes the code much easier to read as well as more convenient to write. As can be found in the section :ref:`Quantumsystems <section-quantumsystems>`, many functions are already implemented that take a basis as argument and generate states and operators that are commonly used in the corresponding quantum systems.

More information on the concrete implementation of states and operators can be found in the section :ref:`States and Operators - Implementation <section-operators-detail>`.



In the following we will assume that

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

* For creating operators of the type :math:`A = I \otimes I \otimes ... a_i ... \otimes I` the very useful embed function can be used:

    :jl:func:`embed(b::Basis, index::Int, op::Operator)`
    :jl:func:`embed(b::Basis, indices::Vector{Int}, ops::Vector{T <: Operator})`

E.g. for a system consisting of 3 spins one can define the basis with::

    b_spin = SpinBasis(1//2)
    b = b_spin ⊗ b_spin ⊗ b_spin

An operator in this basis b that only acts on the second spin could be created as::

    identityoperator(b_spin) ⊗ sigmap(b_spin) ⊗ identityoperator(b_spin)

Equivalently, the embed function simplifies this to::

    embed(b, 2, sigmap(b_spin))
