module particle

import Base.==
import ..operators

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export PositionBasis, MomentumBasis,
        gaussianstate,
        spacing, samplepoints,
        positionoperator, momentumoperator, laplace_x, laplace_p, FFTOperator

"""
Basis for a particle in real space.

For simplicity periodic boundaries are assumed which means that
the rightmost point defined by xmax is not included in the basis
but is defined to be the same as xmin.

Arguments
---------

xmin
    Left boundary.
xmax
    Right boundary.
N
    Number of discrete points in the basis.
"""
type PositionBasis <: Basis
    shape::Vector{Int}
    xmin::Float64
    xmax::Float64
    N::Int
    PositionBasis(xmin::Real, xmax::Real, N::Int) = new([N], xmin, xmax, N)
end


"""
Basis for a particle in momentum space.

For simplicity periodic boundaries are assumed which means that
pmax is not included in the basis but is defined to be the same as pmin.

Arguments
---------

pmin
    Left boundary.
pmax
    Right boundary.
N
    Number of discrete points in the basis.
"""
type MomentumBasis <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    MomentumBasis(pmin::Real, pmax::Real, N::Int) = new([N], pmin, pmax, N)
end


"""
Create a PositionBasis from a MomentumBasis.

Since periodic boundary conditions are assumed the values for :math:`x_{min}`
and :math:`x_{max}` are more or less arbitrary and are chosen to be
:math:`-\\pi/dp` and :math:`\\pi/dp` with :math:`dp=(p_{max}-p_{min})/N`.
"""
PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))

"""
Create a MomentumBasis from a PositionBasis.

Since periodic boundary conditions are assumed the values for :math:`p_{min}`
and :math:`p_{max}` are more or less arbitrary and are chosen to be
:math:`-\\pi/dx` and :math:`\\pi/dx` with :math:`dx=(x_{max}-x_{min})/N`.
"""
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


"""
Create a Gaussian state around x0 and p0 with width sigma in real space.

This implementation is based on the definition

.. math::

    \\Psi(x) = \\frac{\\sqrt{\\Delta x}}{\\pi^{1/4}\\sqrt{\\sigma}}
                e^{i p_0 (x-x_0) - \\frac{(x-x_0)^2}{2 \\sigma^2}}

which has the properties

* :math:`|\\Psi(p)|^2 = 1`
* :math:`\\langle p \\rangle = p_0`
* :math:`\\langle x \\rangle = x_0`
* :math:`\\mathrm{Var}(x) = \\frac{\\sigma^2}{2}`
* :math:`\\mathrm{Var}(p) = \\frac{1}{2 \\sigma^2}`

and is connected to the MomentumBasis based implementation via a
Fourier-transformation

.. math::

    \\Psi(p) = \\frac{1}{\\sqrt{2\\pi}}
                \\int_{-\\infty}^{\\infty} e^{-ipx}\\Psi(x) \\mathrm{d}x

Due to the numerically necessary discretization an additional scaling
factor :math:`\\sqrt{\\Delta x}` is used so that
:math:`\\Psi_i = \\sqrt{\\Delta x} \\Psi(x_i)` and the resulting state
is normalized to

.. math::

    \\sum_i |\\Psi_i|^2 = 1


Arguments
---------
b
    PositionBasis.
x0
    Center of the Gaussian in position space.
p0
    Center of the Gaussian in momentum space.
sigma
    Width of the Gaussian.
"""
function gaussianstate(b::PositionBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dx = spacing(b)
    alpha = 1./(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*(x-x0) - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end


"""
Create a Gaussian state around x0 and p0 with width sigma in momentum space.

This implementation is based on the definition

.. math::

    \\Psi(p) = \\frac{\\sqrt{\\sigma} \\sqrt{\\Delta x}}{\\pi^{1/4}}
                e^{-i x_0 p - \\frac{1}{2}(p-p_0)^2 \\sigma^2}

which has the properties

* :math:`|\\Psi(p)|^2 = 1`
* :math:`\\langle p \\rangle = p_0`
* :math:`\\langle x \\rangle = x_0`
* :math:`\\mathrm{Var}(x) = \\frac{\\sigma^2}{2}`
* :math:`\\mathrm{Var}(p) = \\frac{1}{2 \\sigma^2}`

and is connected to the MomentumBasis based implementation via a
Fourier-transformation

.. math::

    \\Psi(x) = \\frac{1}{\\sqrt{2\\pi}}
                    \\int_{-\\infty}^{\\infty} e^{ipx}\\Psi(p) \\mathrm{d}p

Due to the numerically necessary discretization an additional scaling
factor :math:`\\sqrt{\\Delta x}` is used so that
:math:`\\Psi_i = \\sqrt{\\Delta p} \\Psi(p_i)` and the resulting state
is normalized to

.. math::

    \\sum_i |\\Psi_i|^2 = 1



Arguments
---------
b
    MomentumBasis.
x0
    Center of the Gaussian in position space.
p0
    Center of the Gaussian in momentum space.
sigma
    Width of the Gaussian.
"""
function gaussianstate(b::MomentumBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dp = spacing(b)
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(-1im*x0*p - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end

"""
Distance between two adjacent points of the real space basis.
"""
spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N

"""
Momentum difference between two adjacent points of the momentum basis.
"""
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N

"""
x values of the real space basis.
"""
samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])

"""
p values of the momentum basis.
"""
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

"""
Position operator in real space.
"""
positionoperator(b::PositionBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)), 0, length(b), length(b)))


"""
Position operator in momentum space.
"""
function positionoperator(b::MomentumBasis)
    p_op = DenseOperator(b)
    u = 1im/(12*spacing(b))
    for i=1:b.N-2
        p_op.data[i,i+2] = -u
        p_op.data[i,i+1] = 8*u
        p_op.data[i+1,i] = -8*u
        p_op.data[i+2,i] = u
    end
    return SparseOperator(p_op)
end

"""
Momentum operator in momentum space.
"""
momentumoperator(b::MomentumBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)), 0, length(b), length(b)))

"""
Momentum operator in real space.
"""
function momentumoperator(b::PositionBasis)
    p_op = DenseOperator(b)
    u = -1im/(12*spacing(b))
    for i=1:b.N-2
        p_op.data[i,i+2] = -u
        p_op.data[i,i+1] = 8*u
        p_op.data[i+1,i] = -8*u
        p_op.data[i+2,i] = u
    end
    return SparseOperator(p_op)
end

"""
Second x-derivative in real space.
"""
function laplace_x(b::PositionBasis)
    x_op = DenseOperator(b)
    u = 1/spacing(b)^2
    for i=1:b.N-1
        x_op.data[i+1,i] = u
        x_op.data[i,i] = -2*u
        x_op.data[i,i+1] = u
    end
    x_op.data[b.N,b.N] = -2*u
    return SparseOperator(x_op)
end

"""
Second x-derivative in momentum space.
"""
laplace_x(b::MomentumBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)).^2, 0, length(b), length(b)))

"""
Second p-derivative in momentum space.
"""
function laplace_p(b::MomentumBasis)
    p_op = DenseOperator(b)
    u = 1/spacing(b)^2
    for i=1:b.N-1
        p_op.data[i+1,i] = u
        p_op.data[i,i] = -2*u
        p_op.data[i,i+1] = u
    end
    p_op.data[b.N,b.N] = -2*u
    return SparseOperator(p_op)
end

"""
Second p-derivative in real space.
"""
laplace_p(b::PositionBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)).^2, 0, length(b), length(b)))


type FFTOperator <: Operator
    basis_l::Basis
    basis_r::Basis
    fft_l!
    fft_r!
    mul_before::Vector{Complex128}
    mul_after::Vector{Complex128}
end


"""
DenseOperator performing a fast Fourier transformation of a state.
"""
function FFTOperator(basis_l::MomentumBasis, basis_r::PositionBasis)
    Lx = (basis_r.xmax - basis_r.xmin)
    dp = spacing(basis_l)
    dx = spacing(basis_r)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    x = zeros(Complex128, length(basis_r))
    mul_before = exp(-1im*basis_l.pmin*(samplepoints(basis_r)-basis_r.xmin))
    mul_after = exp(-1im*basis_r.xmin*samplepoints(basis_l))/sqrt(basis_r.N)
    FFTOperator(basis_l, basis_r, plan_bfft!(x), plan_fft!(x), mul_before, mul_after)
end

function FFTOperator(basis_l::PositionBasis, basis_r::MomentumBasis)
    Lx = (basis_l.xmax - basis_l.xmin)
    dp = spacing(basis_r)
    dx = spacing(basis_l)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    x = zeros(Complex128, length(basis_r))
    mul_before = exp(1im*basis_l.xmin*(samplepoints(basis_r)-basis_r.pmin))
    mul_after = exp(1im*basis_r.pmin*samplepoints(basis_l))/sqrt(basis_r.N)
    FFTOperator(basis_l, basis_r, plan_fft!(x), plan_bfft!(x), mul_before, mul_after)
end

operators.full(op::FFTOperator) = op*identityoperator(DenseOperator, op.basis_r)

operators.dagger(op::FFTOperator) = FFTOperator(op.basis_r, op.basis_l)


function operators.gemv!{T<:Complex}(alpha::T, M::FFTOperator, b::Ket, beta::T, result::Ket)
    N::Int = M.basis_r.N
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = M.mul_before[i] * b.data[i]
        end
        M.fft_r! * result.data
        @inbounds for i=1:N
            result.data[i] *= M.mul_after[i] * alpha
        end
    else
        psi_ = Ket(M.basis_l, deepcopy(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= M.mul_before[i]
        end
        M.fft_r! * psi_.data
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * psi_.data[i] * M.mul_after[i]
        end
    end
    nothing
end

function operators.gemv!{T<:Complex}(alpha::T, b::Bra, M::FFTOperator, beta::T, result::Bra)
    N::Int = M.basis_l.N
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = conj(M.mul_after[i]) * conj(b.data[i])
        end
        M.fft_l! * result.data
        @inbounds for i=1:N
            result.data[i] = conj(result.data[i]) * M.mul_before[i] * alpha
        end
    else
        psi_ = Bra(M.basis_r, conj(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= conj(M.mul_after[i])
        end
        M.fft_l! * psi_.data
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * conj(psi_.data[i]) * M.mul_before[i]
        end
    end
    nothing
end


end # module
