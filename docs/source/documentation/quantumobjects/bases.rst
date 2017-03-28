.. _section-bases:

Bases
=====

The primary purpose of bases in **QuantumOptics.jl** is to specify the dimension of the Hilbert space of the system and to make sure that quantum objects associated to distinct bases can't be combined accidentally in an incorrect way. Many of the common types of bases used in quantum mechanics are already implemented in :ref:`section-quantumsystems`.


Composite bases
---------------

Hilbert spaces of composite systems can be handled with the :jl:type:`CompositeBasis` which can be created using the :jl:func:`tensor` function or the equivalent ⊗ operator::

    basis_fock = FockBasis(10)
    basis_particle = MomentumBasis(0., 10., 50)
    basis = tensor(basis_fock, basis_particle)
    basis = basis_fock ⊗ basis_particle

Most of the time this will happen implicitly when tensor products of operators are formed.


Generic bases
-------------

If an appropriate basis type is not implemented, the quick and dirty way is to use a :jl:type:`GenericBasis`, which just needs to know the dimension of the Hilbert space and is ready to go::

    b = GenericBasis(5)

However, since operators and states represented in any generic basis can be combined as long as the bases have the same dimension, it might lead to errors that otherwise would have been caught.


Implementing new bases
----------------------

The cleaner way is to implement own special purpose bases by deriving from the abstract :jl:abstract:`Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of their Hilbert space. E.g. a spin 1/2 basis could be implemented as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        SpinBasis() = new(Int[2]) # Constructor
    end

The default behavior for new bases is to allow operations for bases of the same type, but reject mixing with other bases. Finer control over the interaction with other bases can be achieved by overloading the `==` operator as well as the :jl:func:`multiplicable` function.
