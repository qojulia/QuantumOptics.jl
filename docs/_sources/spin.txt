.. _section-spin:

Spin basis
==========

The spin basis class and all related functions are implemented for arbitrary spin numbers. Therefore, the first step is to choose a basis by specifying the appropriate spin number::

    b = SpinBasis(3//2)

This basis can be used to create operators and states::

    sx = sigmax(b)
    state0 = spinup(b)
    state1 = sx*state0

The definition of the SpinBasis is very simple and is more or less given by::

    type SpinBasis <: Basis
        shape::Vector{Int}
        spinnumber::Rational{Int}
    end

All expected operators are implemented:

* :jl:func:`sigmax`
* :jl:func:`sigmay`
* :jl:func:`sigmaz`
* :jl:func:`sigmap`
* :jl:func:`sigmam`

Also the lowest and uppermost states are defined:

* :jl:func:`spinup`
* :jl:func:`spindown`