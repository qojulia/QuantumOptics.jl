.. _section-manybody:

Many-body system
================

.. code-block:: julia

    b = NLevelBasis(2)
    H = diagonaloperator(b, [0, 1]);

    b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2, 3]))

    H_mb = manybodyoperator(b_mb, H)
    j_mb = transition(b_mb, 1, 2) # Move particles from second to first level
    a1_mb = destroy(b_mb, 1) # Destroy particles in first level
    at2_mb = create(b_mb, 2) # Create particles in second level

Describing systems consisting of many identical particles in a tensor product space created out of single particle Hilbert spaces leads to the problem that not all states in this space correspond to real physical states. In this picture one would have to restrict the Hilbert space to a subspace that is invariant under permutation of particles. However, it is also possible to find a valid description that doesn't first introduce redundant states that later on have to be eliminated. Instead of asking the question in which state a specific particle is, we count how many particles are in this specific state. The specific choice of this states, which is basically a choice of basis, is physically not relevant but nevertheless has to be done. The general idea is to choose a convenient basis :math:`\{\left|u_i\right\rangle\}_i` of the single particle Hilbert space and create the N-particle Hilbert space from states that count how many particles are in each of these states - which will in the following be denoted as :math:`\left|\{n\}\right\rangle`. Of course the sum of these occupation numbers is identical to the number of particles in this state. For fermionic particles an additional restriction is that there can't be more than one particle in one state. There is no reason to restrict ourself to a certain particle number and therefore a single many-body basis can have basis states corresponding to different amount of particles.

This concept is captured in the :jl:type:`ManyBodyBasis` type::

    type ManyBodyBasis <: Basis
        shape::Vector{Int}
        onebodybasis::Basis
        occupations::Vector{Vector{Int}}
    end

Of course we still need a way to relate operators defined in the single particle description to equivalent many-body operators. For an arbitrary additive single particle operator :math:`\sum_i x_i` this connection is given by

.. math::

    X = \sum_{ij} a_i^\dagger a_j
                    \left\langle u_i \right|
                    x
                    \left| u_j \right\rangle

For two particle interactions given by the additive two particle operator :math:`\sum_{i \neq j} V_{ij}` the corresponding many-body operator is given by:

.. math::

    X = \sum_{ijkl} a_i^\dagger a_j^\dagger a_k a_l
            \left\langle u_i \right| \left\langle u_j \right|
            x
            \left| u_k \right\rangle \left| u_l \right\rangle

The :jl:func:`manybodyoperator` function uses these relations to automatically generate many-body operators from their corresponding one-body operators.


States
------

Since it is typically more natural to label a basis state of a many-body basis by its occupation number, e.g. :math:`|2,0,1,0>`, instead of calling it the i-th basis state, the :jl:func:`basisstate` function is overloaded to provide exactly this functionality.

* :jl:func:`basisstate`


Operators
---------

Note that when using the :jl:func:`create` and :jl:func:`destroy` operators for a fixed particle number will most likely fail since every entry of them is zero. Use the :jl:func:`number` and :jl:func:`transition` operators instead.

* :jl:func:`number`
* :jl:func:`create`
* :jl:func:`destroy`
* :jl:func:`transition`

The :jl:func:`manybodyoperator` function allows to automatically generate the many-body operator from the associated one-body operator.

* :jl:func:`manybodyoperator`


Examples
--------

* :ref:`example-manybody-fourlevel-system`
* :ref:`example-nparticles-in-double-well`
