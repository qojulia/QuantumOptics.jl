Building the documentation
==========================

Package requirements
--------------------

The following packages are required:

* Sphinx
* jupyter-notebook
* sphinx-julia::

    >> git clone https://github.com/bastikr/sphinx-julia.git

* pyjulia (Makes sphinx-julia much faster)

    >> git clone https://github.com/JuliaPy/pyjulia.git


Converting notebooks
--------------------

Jupyter notebooks have to be converted to restructuredText

#. Examples:

    >> ./convertexamples.sh

#. Images for tutorial:

    >> ./converttutorial.sh


Prepare build directory
-----------------------

First clone the QuantumOptics.jl repository a second time as ``QuantumOptics.jl-www``. Then change to the gh-pages branch::

    >> git clone git@github.com:bastikr/QuantumOptics.jl.git QuantumOptics.jl-www
    >> cd QuantumOptics.jl-www
    >> git checkout gh-pages

This pulls the website as it is available on https://bastikr.github.io/QuantumOptics.jl/.


Build
-----

Change the current directory to ``QuantumOptics.jl/docs`` and use make::

    >> make html
