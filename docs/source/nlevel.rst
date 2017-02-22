.. _section-nlevel:

N-level basis
=============

Systems consisting of discrete states like e.g. an atom represented by a few relevant levels can be treated using the :jl:type:`NLevelBasis`. The only thing it needs to know is the number of states::

    N = 3
    b = NLevelBasis(N)

Essentially it is defined just as::

    type NLevelBasis <: Basis
        shape::Vector{Int}
        N::Int
    end

We can create a state :math:`|m\rangle` for the level :func:`m` with

* :jl:func:`nlevelstate(b::NLevelBasis, m::Vector{Int})`

With the transition operator, we can create projectors of the form :math:`|m\rangle\langle n|` describing a transition from the state :math:`|n\rangle` to :math:`|n\rangle`.

* :jl:func:`transition(b::NLevelBasis, m::Int, n::Int)`
