.. _section-quantumsystems:

Quantum systems
===============

Building on the :ref:`section-quantumobjects`, **QuantumOptics.jl** provides a convenient way to work with many of the commonly investigated quantum mechanical systems. Primarily, for every such system it defines one or more appropriate bases and allows to quickly create all typically expected operators and states. These then can be used to implement a concrete system.

At the moment the following types of systems are supported:

.. toctree::
    :maxdepth: 1

    spin
    fock
    nlevel
    particle
    subspace
    manybody
