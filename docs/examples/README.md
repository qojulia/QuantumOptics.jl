# QuantumOptics.jl examples

The examples are created as [jupyter notebook](http://jupyter.org/) and converted to markdown with [nbconvert](https://github.com/jupyter/nbconvert).


## Directory layout

When building the documentation the only requirement is that the examples, the documentation and the website are in the same directory:

    |
    |--> ./QuantumOptics.jl-examples
    |--> ./QuantumOptics.jl-documentation
    |--> ./QuantumOptics.jl-website


## Software requirements

* [nbconvert](https://github.com/jupyter/nbconvert)


## Build process

* Make sure the correct version of **QuantumOptics.jl** is in the Julia searchpath.
* Run `julia make.jl`. This will execute the examples and store the markdown output and the generated images in the markdown folder. These files are then copied to `QuantumOptics.jl-documentation/src/examples`.
