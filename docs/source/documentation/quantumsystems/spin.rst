.. _section-spin:

Spin
====

.. code-block:: julia

    b = SpinBasis(3//2)
    psi = spinup(b)
    sx = sigmax(b)

In quantum mechanics particles posses an additional intrinsic property which behaves analogous to classical angular momentum.

Spin systems of arbitrary spin number can be modeled with the :jl:type:`SpinBasis` which is defined as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        spinnumber::Rational{Int}
    end


States
------

The lowest and uppermost states are defined:

* :jl:func:`spinup`
* :jl:func:`spindown`


Operators
---------

All expected operators are implemented, all of which require a single argument of the type SpinBasis :jl:type:`SpinBasis`.

* :jl:func:`sigmax`
* :jl:func:`sigmay`
* :jl:func:`sigmaz`
* :jl:func:`sigmap`
* :jl:func:`sigmam`


Examples
--------

* :ref:`example-jaynes-cummings`
* :ref:`example-two-qubit-entanglement`
