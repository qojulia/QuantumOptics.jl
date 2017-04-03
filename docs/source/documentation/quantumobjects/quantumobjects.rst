.. _section-quantumobjects:

Quantum objects
===============

Quantum objects, i.e. states, operators and super-operators, are the fundamental building blocks used in **QuantumOptics.jl**, which makes it essential to get an intuitive understanding of the underlying concepts of their implementation.

Mathematically, quantum states are abstract vectors in a Hilbert space and operators are linear functions that map these states from one Hilbert space to another. To perform actual numerical calculations with these objects, it is necessary to choose a basis of the Hilbert space under consideration and use their numerical representation relative to this basis. For example if the states :math:`\{|u_i\rangle\}_i` form a basis of a Hilbert space, every possible state :math:`|\psi\rangle` can be expressed as coordinates :math:`\psi_i` in respect to this basis:

.. math::

    |\psi\rangle = \sum_i |u_i\rangle\langle u_i |\psi \rangle
                 = \sum_i \psi_i |u_i\rangle

In **QuantumOptics.jl** all states therefore contain two types of information. The choice of basis and the coefficients of the state with respect to this basis.

Operators are implemented in a very similar fashion. The only thing that makes it more complicated is that in principle it is possible to choose different bases for the left and right hand side, which sometimes is quite useful. Assuming that the states :math:`\{|u_i\rangle\}_{i=1}^{N_u}` form a basis of one Hilbert space :math:`\mathcal{H}_u` and the states :math:`\{|v_i\rangle\}_{i=1}^{N_v}` form a basis of another Hilbert space :math:`\mathcal{H}_v`, every operator defined as map from :math:`\mathcal{H}_v \rightarrow \mathcal{H}_u` can be expressed as coefficients :math:`A_{ij}` in respect to these two bases:

.. math::

    \hat{A} = \sum_{ij} |u_i\rangle\langle u_i| \hat{A} |v_j\rangle\langle v_j|
            = \sum_{ij} A_{ij} |u_i\rangle \langle v_j|

Since for operators two different Hilbert spaces are involved, every operator in **QuantumOptics.jl** has to store information about the left-hand basis, the right-hand basis as well as the coefficients of the operator in respect to these two bases.

For super-operators the number of involved bases further doubles. In general, they map operators defined in :math:`\mathcal{H}_v \rightarrow \mathcal{H}_u` to operators defined in :math:`\mathcal{H}_m \rightarrow \mathcal{H}_n`.

The fact that **QuantumOptics.jl** knows about the choice of basis for every quantum object means that it can check if all performed operations physically make sense, catching many possible mistakes early on. Additionally, explicitly specifying a basis makes the code much easier to read as well as more convenient to write. As can be found in the section :ref:`section-quantumsystems`, many functions are already implemented that take a basis as argument and generate states and operators that are commonly used in the corresponding quantum systems.

More information on the concrete implementation of bases, states, operators and super-operators can be found in the following sections:


.. toctree::
    :maxdepth: 1

    bases
    states
    operators
    superoperators
