Building the documentation
==========================

First clone the QuantumOptics.jl repository a second time as ``QuantumOptics.jl-www``. Then change to the gh-pages branch::

    >> git clone git@github.com:bastikr/QuantumOptics.jl.git QuantumOptics.jl-www
    >> cd QuantumOptics.jl-www
    >> git checkout gh-pages

This pulls the website as it is available on https://bastikr.github.io/QuantumOptics.jl/. To build the documentation one needs the sphinx-julia package::

    >> git clone https://github.com/bastikr/sphinx-julia.git

Then change the current directory to ``QuantumOptics.jl/docs`` and use make::

    >> make html
