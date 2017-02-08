.. _section-particle:

Particle basis
==============

For particles **QuantumOptics.jl** provides two different choices - either the calculations can be done in real space or they can be done in momentum space by using :jl:type:`PositionBasis` or :jl:type:`MomentumBasis` respectively. To create a particle basis, a left boundary :math:`x_\mathrm{min}`, a right boundary :math:`x_\mathrm{max}` and the number of discretization points have to be provided::

    xmin = -2.
    xmax = 4.
    N = 10
    b_position = PositionBasis(xmin, xmax, N)

Equivalently to create a momentum basis a minimal momentum :math:`p_\mathrm{min}`, a maximal momentum :math:`p_\mathrm{min}` and again the number of discretization points have to be specified::

    pmin = 0.
    pmax = 10.
    N = 10
    b_momentum = PositionBasis(pmin, pmax, N)

The definition of these two bases types is very simple::

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

Since real space and momentum space are connected via a Fourier transformation the bases are connected. The numerically inevitable cutoff implies that the functions :math:`\Psi(x)` and :math:`\Psi(p)` can be interpreted to continue periodically over the whole real axis the specific choice of the cutoff points is therefor irrelevant as long as the interval length stays the same. This free choice of cutoff points allows to easily create a corresponding :jl:type:`MomentumBasis` from a :jl:type:`PositionBasis` and vice versa::

    b_momentum = MomentumBasis(b_position)
    b_position = PositionBasis(b_momentum)

When creating a momentum basis from a position basis the cutoff points are connected by :math:`p_\mathrm{min} = -\pi/dx` and :math:`p_\mathrm{max} = \pi/dx` where :math:`dx = (x_\mathrm{max} - x_\mathrm{min})/N`. Similarly for the inverse procedure the cutoffs are :math:`x_\mathrm{min} = -\pi/dp` and :math:`x_\mathrm{max} = \pi/dp` with :math:`dp = (p_\mathrm{max} - p_\mathrm{min})/N`.

For convenience a few functions make it easier to work with bases:

.. epigraph::

    .. jl:autofunction:: particle.jl spacing

    .. jl:autofunction:: particle.jl samplepoints


All operators defined in **QuantumOptics.jl** can be created in respect to both bases, e.g.::

    p_position = momentumoperator(b_position)
    p_momentum = momentumoperator(b_momentum)

The following operators are implemented:

* :jl:func:`momentumoperator(b::PositionBasis)`
* :jl:func:`positionoperator(b::PositionBasis)`
* :jl:func:`laplace_x(b::PositionBasis)`
* :jl:func:`laplace_p(b::PositionBasis)`

And functions for creating states:

* :jl:func:`gaussianstate(b::PositionBasis, , , )`::

    x0 = 0.
    p0 = 1.
    sigma = 2
    Psi_x = gaussianstate(b_position, x0, p0, sigma)
    Psi_p = gaussianstate(b_momentum, x0, p0, sigma)

Transforming a state from one basis into another can be done efficiently using the :jl:type:`FFTOperator` which can be used in the following way::

    op_fft = FFTOperator(basis_momentum, basis_position)
    Psi_p = op_fft*Psi_x

