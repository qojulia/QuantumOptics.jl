module particle

export PositionBasis, MomentumBasis,
        gaussianstate,
        spacing, samplepoints,
        position, momentum, potentialoperator,
        transform

import Base: ==, position
import ..operators

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse


"""
    PositionBasis(xmin, xmax, Npoints)
    PositionBasis(b::MomentumBasis)

Basis for a particle in real space.

For simplicity periodic boundaries are assumed which means that
the rightmost point defined by `xmax` is not included in the basis
but is defined to be the same as `xmin`.

When a [`MomentumBasis`](@ref) is given as argument the exact values
of ``x_{min}`` and ``x_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dp`` and ``\\pi/dp`` with ``dp=(p_{max}-p_{min})/N``.
"""
type PositionBasis <: Basis
    shape::Vector{Int}
    xmin::Float64
    xmax::Float64
    N::Int
    PositionBasis(xmin::Real, xmax::Real, N::Int) = new([N], xmin, xmax, N)
end

"""
    MomentumBasis(pmin, pmax, Npoints)
    MomentumBasis(b::PositionBasis)

Basis for a particle in momentum space.

For simplicity periodic boundaries are assumed which means that
`pmax` is not included in the basis but is defined to be the same as `pmin`.

When a [`PositionBasis`](@ref) is given as argument the exact values
of ``p_{min}`` and ``p_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dx`` and ``\\pi/dx`` with ``dx=(x_{max}-x_{min})/N``.
"""
type MomentumBasis <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
    MomentumBasis(pmin::Real, pmax::Real, N::Int) = new([N], pmin, pmax, N)
end

PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

==(b1::PositionBasis, b2::PositionBasis) = b1.xmin==b2.xmin && b1.xmax==b2.xmax && b1.N==b2.N
==(b1::MomentumBasis, b2::MomentumBasis) = b1.pmin==b2.pmin && b1.pmax==b2.pmax && b1.N==b2.N


"""
    gaussianstate(b::PositionBasis, x0, p0, sigma)
    gaussianstate(b::MomentumBasis, x0, p0, sigma)

Create a Gaussian state around `x0` and` p0` with width `sigma`.

In real space the gaussian state is defined as

```math
\\Psi(x) = \\frac{\\sqrt{\\Delta x}}{\\pi^{1/4}\\sqrt{\\sigma}}
            e^{i p_0 (x-\\frac{x_0}{2}) - \\frac{(x-x_0)^2}{2 \\sigma^2}}
```

and is connected to the momentum space definition

```math
\\Psi(p) = \\frac{\\sqrt{\\sigma} \\sqrt{\\Delta x}}{\\pi^{1/4}}
            e^{-i x_0 (p-\\frac{p_0}{2}) - \\frac{1}{2}(p-p_0)^2 \\sigma^2}
```

via a Fourier-transformation

```math
\\Psi(p) = \\frac{1}{\\sqrt{2\\pi}}
            \\int_{-\\infty}^{\\infty} e^{-ipx}\\Psi(x) \\mathrm{d}x
```

The state has the properties

* ``⟨p⟩ = p_0``
* ``⟨x⟩ = x_0``
* ``\\mathrm{Var}(x) = \\frac{σ^2}{2}``
* ``\\mathrm{Var}(p) = \\frac{1}{2 σ^2}``

Due to the numerically necessary discretization additional scaling
factora ``\\sqrt{Δx}`` and ``\\sqrt{Δp}`` are used so that
``Ψx_i = \\sqrt{Δ x} Ψ(x_i)`` and ``Ψp_i = \\sqrt{Δ x} Ψ(p_i)`` so
that the resulting Ket state is normalized.
"""
function gaussianstate(b::PositionBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dx = spacing(b)
    alpha = 1./(pi^(1/4)*sqrt(sigma))*sqrt(dx)
    x = b.xmin
    for i=1:b.N
        psi.data[i] = alpha*exp(1im*p0*(x-x0/2) - (x-x0)^2/(2*sigma^2))
        x += dx
    end
    return psi
end

function gaussianstate(b::MomentumBasis, x0::Real, p0::Real, sigma::Real)
    psi = Ket(b)
    dp = spacing(b)
    alpha = sqrt(sigma)/pi^(1/4)*sqrt(dp)
    p = b.pmin
    for i=1:b.N
        psi.data[i] = alpha*exp(-1im*x0*(p-p0/2) - (p-p0)^2/2*sigma^2)
        p += dp
    end
    return psi
end


"""
    spacing(b::PositionBasis)

Difference between two adjacent points of the real space basis.
"""
spacing(b::PositionBasis) = (b.xmax - b.xmin)/b.N
"""
    spacing(b::MomentumBasis)

Momentum difference between two adjacent points of the momentum basis.
"""
spacing(b::MomentumBasis) = (b.pmax - b.pmin)/b.N

"""
    samplepoints(b::PositionBasis)

x values of the real space basis.
"""
samplepoints(b::PositionBasis) = (dx = spacing(b); Float64[b.xmin + i*dx for i=0:b.N-1])
"""
    samplepoints(b::MomentumBasis)

p values of the momentum basis.
"""
samplepoints(b::MomentumBasis) = (dp = spacing(b); Float64[b.pmin + i*dp for i=0:b.N-1])

"""
    position(b::PositionBasis)

Position operator in real space.
"""
position(b::PositionBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)), 0, length(b), length(b)))


"""
    position(b:MomentumBasis)

Position operator in momentum space.
"""
function position(b::MomentumBasis)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*full(position(b_pos))*transform(b_pos, b)
end

"""
    momentum(b:MomentumBasis)

Momentum operator in momentum space.
"""
momentum(b::MomentumBasis) = SparseOperator(b, spdiagm(complex(samplepoints(b)), 0, length(b), length(b)))

"""
    momentum(b::PositionBasis)

Momentum operator in real space.
"""
function momentum(b::PositionBasis)
    b_mom = MomentumBasis(b)
    transform(b, b_mom)*full(momentum(b_mom))*transform(b_mom, b)
end

"""
    potentialoperator(b::PositionBasis, V(x))

Operator representing a potential ``V(x)`` in real space.
"""
function potentialoperator(b::PositionBasis, V::Function)
    x = samplepoints(b)
    diagonaloperator(b, V.(x))
end

"""
    potentialoperator(b::MomentumBasis, V(x))

Operator representing a potential ``V(x)`` in momentum space.
"""
function potentialoperator(b::MomentumBasis, V::Function)
    b_pos = PositionBasis(b)
    transform(b, b_pos)*full(potentialoperator(b_pos, V))*transform(b_pos, b)
end

"""
    FFTOperator

Abstract type for all implementations of FFT operators.
"""
abstract type FFTOperator <: Operator end

PlanFFT = Base.DFT.FFTW.cFFTWPlan

"""
    FFTOperators

Operator performing a fast fourier transformation when multiplied with a state
that is a Ket or an Operator.
"""
type FFTOperators <: FFTOperator
    basis_l::Basis
    basis_r::Basis
    fft_l!::PlanFFT
    fft_r!::PlanFFT
    fft_l2!::PlanFFT
    fft_r2!::PlanFFT
    mul_before::Array{Complex128}
    mul_after::Array{Complex128}
end

"""
    FFTKets

Operator that can only perform fast fourier transformations on Kets.
This is much more memory efficient when only working with Kets.
"""
type FFTKets <: FFTOperator
    basis_l::Basis
    basis_r::Basis
    fft_l!::PlanFFT
    fft_r!::PlanFFT
    mul_before::Array{Complex128}
    mul_after::Array{Complex128}
end

"""
    transform(b1::MomentumBasis, b2::PositionBasis)
    transform(b1::PositionBasis, b2::MomentumBasis)

Transformation operator between position basis and momentum basis.
"""
function transform(basis_l::MomentumBasis, basis_r::PositionBasis; ket_only::Bool=false)
    Lx = (basis_r.xmax - basis_r.xmin)
    dp = spacing(basis_l)
    dx = spacing(basis_r)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(-1im*basis_l.pmin*(samplepoints(basis_r)-basis_r.xmin))
    mul_after = exp.(-1im*basis_r.xmin*samplepoints(basis_l))/sqrt(basis_r.N)
    x = Vector{Complex128}(length(basis_r))
    if ket_only
        FFTKets(basis_l, basis_r, plan_bfft!(x), plan_fft!(x), mul_before, mul_after)
    else
        A = Matrix{Complex128}(length(basis_r), length(basis_r))
        FFTOperators(basis_l, basis_r, plan_bfft!(x), plan_fft!(x), plan_bfft!(A, 2), plan_fft!(A, 1), mul_before, mul_after)
    end
end

"""
    transform(b1::CompositeBasis, b2::CompositeBasis)

Transformation operator between two composite bases. Each of the bases
has to contain bases of type PositionBasis and the other one a corresponding
MomentumBasis.
"""
function transform(basis_l::PositionBasis, basis_r::MomentumBasis; ket_only::Bool=false)
    Lx = (basis_l.xmax - basis_l.xmin)
    dp = spacing(basis_r)
    dx = spacing(basis_l)
    if basis_l.N != basis_r.N || abs(2*pi/dp - Lx)/Lx > 1e-12
        throw(IncompatibleBases())
    end
    mul_before = exp.(1im*basis_l.xmin*(samplepoints(basis_r)-basis_r.pmin))
    mul_after = exp.(1im*basis_r.pmin*samplepoints(basis_l))/sqrt(basis_r.N)
    x = Vector{Complex128}(length(basis_r))
    if ket_only
        FFTKets(basis_l, basis_r, plan_fft!(x), plan_bfft!(x), mul_before, mul_after)
    else
        A = Matrix{Complex128}(length(basis_r), length(basis_r))
        FFTOperators(basis_l, basis_r, plan_fft!(x), plan_bfft!(x), plan_fft!(A, 2), plan_bfft!(A, 1), mul_before, mul_after)
    end
end

function transform(basis_l::CompositeBasis, basis_r::CompositeBasis; ket_only::Bool=false, index::Vector{Int}=Int[])
    @assert length(basis_l.bases) == length(basis_r.bases)
    if length(index) == 0
        check_pos = typeof.(basis_l.bases) .== PositionBasis
        check_mom = typeof.(basis_l.bases) .== MomentumBasis
        if any(check_pos) && !any(check_mom)
            index = [1:length(basis_l.bases);][check_pos]
        elseif any(check_mom) && !any(check_pos)
            index = [1:length(basis_l.bases);][check_mom]
        else
            throw(IncompatibleBases())
        end
    end
    if all(typeof.(basis_l.bases[index]) .== PositionBasis)
        @assert all(typeof.(basis_r.bases[index]) .== MomentumBasis)
        transform_xp(basis_l, basis_r, index; ket_only=ket_only)
    elseif all(typeof.(basis_l.bases[index]) .== MomentumBasis)
        @assert all(typeof.(basis_r.bases[index]) .== PositionBasis)
        transform_px(basis_l, basis_r, index; ket_only=ket_only)
    else
        throw(IncompatibleBases())
    end
end

function transform_xp(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Vector{Int}; ket_only::Bool=false)
    n = length(basis_l.bases)
    Lx = [(b.xmax - b.xmin) for b=basis_l.bases[index]]
    dp = [spacing(b) for b=basis_r.bases[index]]
    dx = [spacing(b) for b=basis_l.bases[index]]
    N = [length(b) for b=basis_l.bases]
    for i=1:n
        if N[i] != length(basis_r.bases[i])
            throw(IncompatibleBases())
        end
    end
    for i=1:length(index)
        if abs(2*pi/dp[i] - Lx[i])/Lx[i] > 1e-12
            throw(IncompatibleBases())
        end
    end

    if index[1] == 1
        mul_before = exp.(1im*basis_l.bases[1].xmin*(samplepoints(basis_r.bases[1])-basis_r.bases[1].pmin))
        mul_after = exp.(1im*basis_r.bases[1].pmin*samplepoints(basis_l.bases[1]))/sqrt(basis_r.bases[1].N)
    else
        mul_before = ones(N[1])
        mul_after = ones(N[1])
    end
    for i=2:n
        if any(i .== index)
            mul_before = kron(exp.(1im*basis_l.bases[i].xmin*(samplepoints(basis_r.bases[i])-basis_r.bases[i].pmin)), mul_before)
            mul_after = kron(exp.(1im*basis_r.bases[i].pmin*samplepoints(basis_l.bases[i]))/sqrt(basis_r.bases[i].N), mul_after)
        else
            mul_before = kron(ones(N[i]), mul_before)
            mul_after = kron(ones(N[i]), mul_after)
        end
    end
    mul_before = reshape(mul_before, (N...))
    mul_after = reshape(mul_after, (N...))

    x = Array{Complex128}(N...)
    if ket_only
        FFTKets(basis_l, basis_r, plan_fft!(x, index), plan_bfft!(x, index), mul_before, mul_after)
    else
        A = Array{Complex128}([N, N;]...)
        FFTOperators(basis_l, basis_r, plan_fft!(x, index), plan_bfft!(x, index), plan_fft!(A, [n + 1:2n;][index]), plan_bfft!(A, [1:n;][index]), mul_before, mul_after)
    end
end

function transform_px(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Vector{Int}; ket_only::Bool=false)
    n = length(basis_l.bases)
    Lx = [(b.xmax - b.xmin) for b=basis_r.bases[index]]
    dp = [spacing(b) for b=basis_l.bases[index]]
    dx = [spacing(b) for b=basis_r.bases[index]]
    N = [length(b) for b=basis_l.bases]
    for i=1:n
        if N[i] != length(basis_r.bases[i])
            throw(IncompatibleBases())
        end
    end
    for i=1:length(index)
        if abs(2*pi/dp[i] - Lx[i])/Lx[i] > 1e-12
            throw(IncompatibleBases())
        end
    end

    if index[1] == 1
        mul_before = exp.(-1im*basis_l.bases[1].pmin*(samplepoints(basis_r.bases[1])-basis_r.bases[1].xmin))
        mul_after = exp.(-1im*basis_r.bases[1].xmin*samplepoints(basis_l.bases[1]))/sqrt(N[1])
    else
        mul_before = ones(N[1])
        mul_after = ones(N[1])
    end
    for i=2:n
        if i in index
            mul_before = kron(exp.(-1im*basis_l.bases[i].pmin*(samplepoints(basis_r.bases[i])-basis_r.bases[i].xmin)), mul_before)
            mul_after = kron(exp.(-1im*basis_r.bases[i].xmin*samplepoints(basis_l.bases[i]))/sqrt(N[i]), mul_after)
        else
            mul_before = kron(ones(N[i]), mul_before)
            mul_after = kron(ones(N[i]), mul_after)
        end
    end
    mul_before = reshape(mul_before, (N...))
    mul_after = reshape(mul_after, (N...))

    x = Array{Complex128}(N...)
    if ket_only
        FFTKets(basis_l, basis_r, plan_bfft!(x, index), plan_fft!(x, index), mul_before, mul_after)
    else
        A = Array{Complex128}([N, N;]...)
        FFTOperators(basis_l, basis_r, plan_bfft!(x, index), plan_fft!(x, index), plan_bfft!(A, [n + 1:2n;][index]), plan_fft!(A, [1:n;][index]), mul_before, mul_after)
    end
end

operators.full(op::FFTOperators) = op*identityoperator(DenseOperator, op.basis_r)

operators.dagger(op::FFTOperators) = transform(op.basis_r, op.basis_l)
operators.dagger(op::FFTKets) = transform(op.basis_r, op.basis_l; ket_only=true)

operators.tensor(A::FFTOperators, B::FFTOperators) = transform(tensor(A.basis_l, B.basis_l), tensor(A.basis_r, B.basis_r))
operators.tensor(A::FFTKets, B::FFTKets) = transform(tensor(A.basis_l, B.basis_l), tensor(A.basis_r, B.basis_r); ket_only=true)

function operators.gemv!(alpha_, M::FFTOperator, b::Ket, beta_, result::Ket)
    alpha = convert(Complex128, alpha_)
    beta = convert(Complex128, beta_)
    N::Int = length(M.basis_r)
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = M.mul_before[i] * b.data[i]
        end
        M.fft_r! * reshape(result.data, size(M.mul_before))
        @inbounds for i=1:N
            result.data[i] *= M.mul_after[i] * alpha
        end
    else
        psi_ = Ket(M.basis_l, copy(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= M.mul_before[i]
        end
        M.fft_r! * reshape(psi_.data, size(M.mul_before))
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * psi_.data[i] * M.mul_after[i]
        end
    end
    nothing
end

function operators.gemv!(alpha_, b::Bra, M::FFTOperator, beta_, result::Bra)
    alpha = convert(Complex128, alpha_)
    beta = convert(Complex128, beta_)
    N::Int = length(M.basis_l)
    if beta==Complex(0.)
        @inbounds for i=1:N
            result.data[i] = conj(M.mul_after[i]) * conj(b.data[i])
        end
        M.fft_l! * reshape(result.data, size(M.mul_after))
        @inbounds for i=1:N
            result.data[i] = conj(result.data[i]) * M.mul_before[i] * alpha
        end
    else
        psi_ = Bra(M.basis_r, conj(b.data))
        @inbounds for i=1:N
            psi_.data[i] *= conj(M.mul_after[i])
        end
        M.fft_l! * reshape(psi_.data, size(M.mul_after))
        @inbounds for i=1:N
            result.data[i] = beta*result.data[i] + alpha * conj(psi_.data[i]) * M.mul_before[i]
        end
    end
    nothing
end

function operators.gemm!(alpha_, A::DenseOperator, B::FFTOperators, beta_, result::DenseOperator)
    alpha = convert(Complex128, alpha_)
    beta = convert(Complex128, beta_)
    if beta != Complex(0.)
        data = Matrix{Complex128}(size(result.data, 1), size(result.data, 2))
    else
        data = result.data
    end
    copy!(data, A.data)
    scale!(data, B.mul_after[:])
    conj!(data)
    B.fft_l2! * reshape(data, [size(B.mul_after)..., size(B.mul_after)...;]...)
    conj!(data)
    scale!(data, B.mul_before[:])
    if alpha != Complex(1.)
        scale!(alpha, data)
    end
    if beta != Complex(0.)
        scale!(result.data, beta)
        result.data += data
    end
    nothing
end

function operators.gemm!(alpha_, A::FFTOperators, B::DenseOperator, beta_, result::DenseOperator)
    alpha = convert(Complex128, alpha_)
    beta = convert(Complex128, beta_)
    if beta != Complex(0.)
        data = Matrix{Complex128}(size(result.data, 1), size(result.data, 2))
    else
        data = result.data
    end
    copy!(data, B.data)
    scale!(A.mul_before[:], data)
    A.fft_r2! * reshape(data, [size(A.mul_before)..., size(A.mul_before)...;]...)
    scale!(A.mul_after[:], data)
    if alpha != Complex(1.)
        scale!(alpha, data)
    end
    if beta != Complex(0.)
        scale!(result.data, beta)
        result.data += data
    end
    nothing
end


end # module
