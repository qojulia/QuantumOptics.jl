.. _section-bases:

Bases
=====

The primary purpose of bases in **QuantumOptics.jl** is to specify the dimension of the Hilbert space of the system and to make sure that quantum objects associated to distinct bases can't be combined accidentally in an incorrect way. Many of the common types of bases used in quantum mechanics like

* :ref:`Spin basis <section-spin>`
* :ref:`Fock basis <section-fock>`
* :ref:`N-level basis <section-nlevel>`
* :ref:`Position basis and Momentum basis <section-particle>`
* :ref:`N-particle basis <section-nparticles>`

are already implemented. They are treated in more detail in the section :ref:`section-quantumsystems`.


Composite bases
---------------

Hilbert spaces of composite systems can be handled with the :jl:type:`CompositeBasis` which can be created using the `tensor` function or the equivalent ⊗ operator::

    basis_fock = FockBasis(10)
    basis_particle = MomentumBasis(0., 10., 50)
    basis = tensor(basis_fock, basis_particle)
    basis = basis_fock ⊗ basis_particle

Most of the time this will happen implicitly when operators are combined using the tensor function.

Subspace basis
--------------

Restricting a Hilbert space to a subspace is done using a :jl:type:`SubspaceBasis`. It is defined by :math:`N`, not necessarily orthogonal states :math:`\{|u\rangle\}` that live in the embedding Hilbert space. However, for the following operations to work correctly, the basis states have to be orthonormal. This can be achieved for any SubspaceBasis with help of the :jl:func:`orthonormalize` function which utilizes the numerical stable modified Gram-Schmidt algorithm. Projecting a state :math:`|x\rangle` into the subspace,

.. math::

    | x^\prime \rangle
            = \sum_{u \in \mathrm{subspace}} |u \rangle \langle u | x \rangle

results in the state :math:`|x^\prime\rangle`. This is done with a projection operator that can be obtained via the :jl:func:`projector(::SubspaceBasis, ::Basis)` function::

    b = FockBasis(5)
    b_sub = SubspaceBasis(b, [fockstate(b, 1), fockstate(b, 2)])

    P = projector(b_sub, b)

    x = coherentstate(b, 0.5)
    x_prime = P*x

The projection operation is irreversible if the original state was not already contained in the subspace. However, it is of course possible to represent any state contained in the subspace in the superbasis::

    y = dagger(P)*x_prime # Not equal to x


Generic bases
-------------

If a needed basis type is not implemented the quick and dirty way is to use a :jl:type:`GenericBasis`, which just needs to know the dimension of the Hilbert space and is ready to go::

    b = GenericBasis(5)

However, since operators and states represented in any generic basis can be combined as long as the bases have the same dimension it might lead to errors that otherwise could have been caught easily.


Implementing new bases
----------------------

The cleaner way is to implement own special purpose bases by deriving from the abstract :jl:abstract:`Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of their Hilbert space. E.g. a spin 1/2 basis could be implemented as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        SpinBasis() = new(Int[2]) # Constructor
    end

The default behavior for new bases is to allow operations for bases of the same type, but reject mixing with other bases. Finer control over the interaction with other bases can be achieved by overloading the `==` operator as well as the :jl:func:`bases.multiplicable` function.
