.. _section-subspace:

Subspace
========

.. code-block:: julia

    b = FockBasis(5)
    b_sub = SubspaceBasis(b, [fockstate(b, 1), fockstate(b, 2)])

    P = projector(b_sub, b)

    x = coherentstate(b, 0.5)
    x_prime = P*x
    y = dagger(P)*x_prime # Not equal to x

Oftentimes it is possible to restrict a large Hilbert to a small subspace while still retaining the most important physical effects. This reduction can be done with the :jl:type:`SubspaceBasis` which is implemented as::

    type SubspaceBasis <: Basis
        shape::Vector{Int}
        superbasis::Basis
        basisstates::Vector{Ket}


Operators
---------

To project states into the subspace or re-embed them into the super-space a projection operator can be used:

* :jl:func:`projector`


Additional functions
--------------------

If the states used to define the subspace are not orthonormal one can use the orthonormalize function to obtain an ONB:

* :jl:func:`orthonormalize`


Examples
--------

* :ref:`example-nparticles-in-double-well`
