.. _section-fock:

Fock Basis
==========

To create a basis of a Fock space **QuantumOptics.jl** provides the :jl:type:`FockBasis` class which has to be supplied with an upper cutoff and optionally with a lower cutoff::

    Nmax = 10
    b1 = FockBasis(Nmax)

    Nmin = 2
    Nmax = 12
    b2 = FockBasis(Nmin, Nmax)

In this example even though the dimensions of the Hilbert spaces described by these bases are the same ``b1`` and ``b2`` are not and mixing operators in one bases with operators in the other basis will result in an explicit error.

The definition of :jl:type:`FockBasis` is essentially::

    type FockBasis <: Basis
        shape::Vector{Int}
        Nmin::Int
        Nmax::Int
    end

Many common operators are already defined:

* :jl:func:`number`
* :jl:func:`destroy`
* :jl:func:`create`

Fock states and coherent states can be created using the functions:

* :jl:func:`fockstate`
* :jl:func:`coherentstate`

Additional functions:

.. epigraph::

    .. jl:autofunction:: fock.jl qfunc(rho, alpha)
