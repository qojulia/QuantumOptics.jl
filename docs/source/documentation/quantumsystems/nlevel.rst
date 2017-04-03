.. _section-nlevel:

N-Level system
==============

.. code-block:: julia

    N = 3
    b = NLevelBasis(N)
    psi2 = nlevelstate(b, 2)
    psi1 = transition(b, 1, 2)

N-Level systems are mostly used as idealized model when the relevant physics can be captured in a small subspace of the complete Hilbert space. For example an atom often can be represented by a few relevant levels.

For these kinds of systems the :jl:type:`NLevelBasis` can be used. Essentially it is defined just as::

    type NLevelBasis <: Basis
        shape::Vector{Int}
        N::Int
    end


States
------

We can create a state :math:`|m\rangle` with

:jl:func:`nlevelstate(b::NLevelBasis, m::Int)`


Operators
---------

With the transition operator, we can create projectors of the form :math:`|m\rangle\langle n|` describing a transition from the state :math:`|n\rangle` to :math:`|m\rangle`.

:jl:func:`transition(b::NLevelBasis, m::Int, n::Int)`


Examples
--------

* :ref:`example-raman`
* :ref:`example-manybody-fourlevel-system`
