.. _section-fock:

Fock Basis
==========

.. code-block:: julia

    N = 10
    b = FockBasis(N)
    alpha = 0.4
    psi = coherentstate(b, alpha)
    a = destroy(b)

A fock space describes the situation of variable particle number. I.e. the system can have zero particles, one particle, two particles and so on.

To create a basis of a Fock space **QuantumOptics.jl** provides the :jl:type:`FockBasis` class which has to be supplied with an integer specifying the maximum number of photons. It is defined as::

    type FockBasis <: Basis
        shape::Vector{Int}
        Nmin::Int
        Nmax::Int
    end


States
------

* :jl:func:`fockstate`
* :jl:func:`coherentstate`


Operators
---------

* :jl:func:`number`
* :jl:func:`destroy`
* :jl:func:`create`


Additional functions
--------------------

* :jl:func:`qfunc`


Examples
--------

* :ref:`example-pumped-cavity`
* :ref:`example-jaynes-cummings`
* :ref:`example-correlation-spectrum`