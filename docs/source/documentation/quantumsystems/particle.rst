.. _section-particle:

Particle
========

.. code-block:: julia

    xmin = -2.
    xmax = 4.
    N = 10
    b_position = PositionBasis(xmin, xmax, N)
    b_momentum = MomentumBasis(b_position)

    x0 = 1.2
    p0 = 0.4
    sigma = 0.2
    psi = gaussianstate(b_position, x0, p0, sigma)

    x = position(b_position)
    p = momentum(b_position)


For particles **QuantumOptics.jl** provides two different choices - either the calculations can be done in real space or they can be done in momentum space by using :jl:type:`PositionBasis` or :jl:type:`MomentumBasis` respectively. The definition of these two bases types is::

    type PositionBasis <: Basis
        shape::Vector{Int}
        xmin::Float64
        xmax::Float64
        N::Int
    end

    type MomentumBasis <: Basis
        shape::Vector{Int}
        pmin::Float64
        pmax::Float64
        N::Int
    end

Since real space and momentum space are connected via a Fourier transformation the bases themselfes are connected. The numerically inevitable cutoff implies that the functions :math:`\Psi(x)` and :math:`\Psi(p)` can be interpreted to continue periodically over the whole real axis. The specific choice of the cutoff points is therefore irrelevant as long as the interval length stays the same. This free choice of cutoff points allows to easily create a corresponding :jl:type:`MomentumBasis` from a :jl:type:`PositionBasis` and vice versa::

    b_momentum = MomentumBasis(b_position)
    b_position = PositionBasis(b_momentum)

When creating a momentum basis from a position basis the cutoff points are connected by :math:`p_\mathrm{min} = -\pi/dx` and :math:`p_\mathrm{max} = \pi/dx` where :math:`dx = (x_\mathrm{max} - x_\mathrm{min})/N`. Similarly for the inverse procedure the cutoffs are :math:`x_\mathrm{min} = -\pi/dp` and :math:`x_\mathrm{max} = \pi/dp` with :math:`dp = (p_\mathrm{max} - p_\mathrm{min})/N`.


States
------

* :jl:func:`gaussianstate`


Operators
---------

All operators are defined for the position basis as well as for the momentum basis.

* :jl:func:`momentum`
* :jl:func:`position`

Transforming a state from one basis into another can be done efficiently using the :jl:type:`FFTOperator`::

    Tpx = FFTOperator(basis_momentum, basis_position)
    Psi_p = Tpx*Psi_x


Additional functions
--------------------

* :jl:func:`spacing`
* :jl:func:`samplepoints`


Examples
--------

* :ref:`example-particle-in-harmonic-trap`
* :ref:`example-particle-into-barrier`
* :ref:`example-nparticles-in-double-well`
