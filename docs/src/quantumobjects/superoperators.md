# Super-operators

Let's now consider mappings from the space of mappings ``\mathcal{H} \rightarrow \mathcal{H}`` to itself, i.e. ``(\mathcal{H} \rightarrow \mathcal{H}) \rightarrow (\mathcal{H} \rightarrow \mathcal{H})``. In operator notation we also call these objects *super-operators*. With the operators ``A,B`` and the super-operator ``S`` the basis independent expression is denoted by

```math
A = S B
```

In contrast, for the basis specific version we have to choose two possibly different bases for A which we denote as ``\{|u\rangle\}`` and ``\{|v\rangle\}`` and additionally two, also possibly different bases for B, ``\{|m\rangle\}`` and ``\{|n\rangle\}``.

```math
A &= \sum_{uv} A_{uv} |v \rangle \langle u|
\\
B &= \sum_{mn} B_{mn} |n \rangle \langle m|
\\
S &= \sum_{uvmn} S_{uvmn} |v \rangle \langle u| \otimes
                          |n \rangle \langle m|
```

The coefficients are then connected by

```math
A_{uv} = \sum_{mn} S_{uvmn} B_{mn}
```

The implementation of super-operators in **QuantumOptics.jl** therefore has to know about four, possibly different, bases. The two basis choices for the codomain (output) are stored in the `basis_l` field and the two basis choices for the domain (input) are stored in the `basis_r` field. At the moment there is one concrete [`SuperOperator`](@ref) type implemented.

Besides the expected algebraic operations there are a few additional functions that help creating and working with super-operators:

* [`spre`](@ref)
* [`spost`](@ref)
* [`liouvillian`](@ref)
* [`exp(::SuperOperator)`](@ref)
