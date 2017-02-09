Super-operators
===============

If states are defined as abstract elements of the Hilbert space :math:`\mathcal{H}` then operators are mappings from this Hilbert space to itself, :math:`\mathcal{H} \rightarrow \mathcal{H}`. However, in **QuantumOptics.jl** states are specified as coefficients in respect to a specific basis and therefore operators are mappings from elements of the Hilbert space in a certain basis to a elements of the same Hilbert space but possibly in a different basis. The basis free definition is

.. math::

    |\Psi\rangle = A |\Phi\rangle

while for a basis specific version we have to choose two possibly different bases :math:`\{|u\rangle\}` and :math:`\{|v\rangle\}` and express the states :math:`|\Psi\rangle` and :math:`|\Phi\rangle` and the operator A in these bases

.. math::

    |\Psi\rangle &= \sum_u \Psi_u \langle u |\Psi\rangle
    \\
    |\Phi\rangle &= \sum_v \Phi_v \langle v |\Phi\rangle
    \\
    A =  \sum_{uv} A_{uv} | v \rangle \langle u |

The coefficients are then connected by the equation

.. math::

    \Psi_u = \sum_v A_{uv} \Phi_v

As next level we now consider mappings from the space of mappings :math:`\mathcal{H} \rightarrow \mathcal{H}` to itself, i.e. :math:`(\mathcal{H} \rightarrow \mathcal{H}) \rightarrow (\mathcal{H} \rightarrow \mathcal{H})`. In operator notation we also call these objects *super-operators*. With the operators :math:`A,B` and the super-operator :math:`S` the basis independent expression is denoted by

.. math::

    A = S B

In contrast, for the basis specific version we have to choose two possibly different bases for A which we denote as :math:`\{|u\rangle\}` and :math:`\{|v\rangle\}` and additionally two, also possibly different bases for B, :math:`\{|m\rangle\}` and :math:`\{|n\rangle\}`.

.. math::

    A &= \sum_{uv} A_{uv} |v \rangle \langle u|
    \\
    B &= \sum_{mn} B_{mn} |n \rangle \langle m|
    \\
    S &= \sum_{uvmn} S_{uvmn} |v \rangle \langle u| \otimes
                              |n \rangle \langle m|

The coefficients are then connected by

.. math::

    A_{uv} &= \sum_{mn} S_{uvmn} B_{mn}

The implementation of super-operators in **QuantumOptics.jl** is based on the basis specific concept, which means it has to consider 4 possibly different bases. The two basis choices for the output are stored in the ``basis_l`` field and the two basis choices for the input are stored in the ``basis_r`` field. At the moment there are two concrete super-operator types implemented, a dense version :jl:type:`DenseSuperOperator` and a sparse version :jl:type:`SparseSuperOperator`, both inheriting from the abstract :jl:abstract:`SuperOperator` type.

Besides the expected algebraic operations there are a few additional functions that help creating and working with super-operators:

* :jl:func:`spre(::DenseOperator)`
* :jl:func:`spost(::DenseOperator)`
* :jl:func:`liouvillian(H, J)`
* :jl:func:`expm`