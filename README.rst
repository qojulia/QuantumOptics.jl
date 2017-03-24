QuantumOptics.jl
================

**QuantumOptics.jl** is a numerical framework written in `Julia <http://julialang.org/>`_ that makes it easy to simulate various kinds of quantum systems. It is similar to the `Quantum Optics Toolbox <http://qo.phy.auckland.ac.nz/toolbox/>`_ for MATLAB and its Python equivalent `QuTiP <http://qutip.org/>`_.

.. image:: https://api.travis-ci.org/bastikr/QuantumOptics.jl.png?branch=master
   :alt: Travis build status
   :target: https://travis-ci.org/bastikr/QuantumOptics.jl

.. image:: https://ci.appveyor.com/api/projects/status/t83f2bqfpumn6d96/branch/master?svg=true
   :alt: Windows build status
   :target: https://ci.appveyor.com/project/bastikr/quantumoptics-jl/branch/master

.. image:: https://coveralls.io/repos/github/bastikr/QuantumOptics.jl/badge.svg?branch=master
   :alt: Test coverage status on coveralls
   :target: https://coveralls.io/github/bastikr/QuantumOptics.jl?branch=master

.. image:: https://codecov.io/gh/bastikr/QuantumOptics.jl/branch/master/graph/badge.svg
   :alt: Test coverage status on codecov
   :target: https://codecov.io/gh/bastikr/QuantumOptics.jl


Example
-------

.. code-block:: julia

    using QuantumOptics

    b = SpinBasis(1//2)
    H = sigmap(b) + sigmam(b)
    psi0 = spindown(b)

    T = [0:0.1:1;]
    tout, psit = timeevolution.schroedinger(T, psi0, H)

More involved examples created using jupyter notebooks can be found at

    https://bastikr.github.io/QuantumOptics.jl/docs/examples.html


Documentation
-------------

The documentation written with `Sphinx <http://www.sphinx-doc.org/>`_ using the `Sphinx-Julia <https://github.com/bastikr/sphinx-julia>`_ plugin is available at

    https://bastikr.github.io/QuantumOptics.jl/docs
