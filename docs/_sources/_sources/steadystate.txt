.. _section-steadystate:

Steady state
============

**QuantumOptics.jl** implements two different ways to calculate steady states. The first one is to perform a time evolution according to a master equation until a adequate accuracy is reached:

.. epigraph::

    .. jl:autofunction:: steadystate.jl master

For smaller system sizes finding eigenvectors of super-operators is the prefered method:

.. epigraph::

    .. jl:autofunction:: steadystate.jl eigenvector(L::DenseSuperOperator)

    .. jl:autofunction:: steadystate.jl eigenvector(H, J)
