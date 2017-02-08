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

Defined operators:

* :jl:func:`transition`


States can be created with

* :jl:func:`nlevelstate`
