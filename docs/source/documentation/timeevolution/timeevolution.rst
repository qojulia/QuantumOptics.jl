.. _section-timeevolution:

Time-evolution
==============

**QuantumOptics.jl** implements various solver to simulate the dynamics of closed as well as open quantum systems. The interfaces are designed to be as consistent as possible which makes it easy to switch between different methods.

.. code-block:: julia

    tout, psi_t = timeevolution.schroedinger(T, psi0, H)
    tout, rho_t = timeevolution.master(T, psi0, H, J)
    tout, psi_t = timeevolution.mcwf(T, psi0, H, J)

More details can be found in the following sections:

.. toctree::
    :maxdepth: 1

    schroedinger
    master
    mcwf
