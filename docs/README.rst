Building the documentation
==========================

First clone the QuantumOptics.jl repository a second time and name it ``QuantumOptics.jl.git-docs``. Then change to the gh-pages branch::

    >> git clone git@github.com:bastikr/QuantumOptics.jl.git QuantumOptics.jl-www
    >> cd QuantumOptics.jl-www
    >> git checkout gh-pages

This pulls the website as it is available on https://bastikr.github.io/QuantumOptics.jl/.
To build the latest documentation change the current directory to ``QuantumOptics.jl/docs`` and use make::

    >> make html