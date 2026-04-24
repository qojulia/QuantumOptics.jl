# Particle

```@example particle
using QuantumOptics # hide
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
nothing # hide
```

For particles **QuantumOptics.jl** provides two different choices - either the calculations can be done in real space or they can be done in momentum space by using [`PositionBasis`](@ref) or [`MomentumBasis`](@ref) respectively. The definition of these two bases types is:

```julia
struct PositionBasis <: Basis
    shape::Vector{Int}
    xmin::Float64
    xmax::Float64
    N::Int
end

struct MomentumBasis <: Basis
    shape::Vector{Int}
    pmin::Float64
    pmax::Float64
    N::Int
end
```

Since real space and momentum space are connected via a Fourier transformation the bases themselves are connected. The numerically inevitable cutoff implies that the functions ``\Psi(x)`` and ``\Psi(p)`` can be interpreted to continue periodically over the whole real axis. The specific choice of the cutoff points is therefore irrelevant as long as the interval length stays the same. This free choice of cutoff points allows to easily create a corresponding [`MomentumBasis`](@ref) from a [`PositionBasis`](@ref) and vice versa:

```@example particle
b_momentum = MomentumBasis(b_position)
b_position = PositionBasis(b_momentum)
nothing # hide
```

When creating a momentum basis from a position basis the cutoff points are connected by ``p_\mathrm{min} = -\pi/dx`` and ``p_\mathrm{max} = \pi/dx`` where ``dx = (x_\mathrm{max} - x_\mathrm{min})/N``. Similarly for the inverse procedure the cutoffs are ``x_\mathrm{min} = -\pi/dp`` and ``x_\mathrm{max} = \pi/dp`` with ``dp = (p_\mathrm{max} - p_\mathrm{min})/N``.


## [States](@id particle: States)

* [`gaussianstate`](@ref)


## [Operators](@id particle: Operators)

All operators are defined for the position basis as well as for the momentum basis.

* [`momentum`](@ref)
* [`position`](@ref)

Transforming a state from one basis into another can be done efficiently using the [`transform`](@ref):

    Tpx = transform(basis_momentum, basis_position)
    Psi_p = Tpx*Psi_x

This also works for composite bases, so long as at least one of them is a [`PositionBasis`](@ref) or [`MomentumBasis`](@ref), respectively. For example:

```@example particle
b_fock = FockBasis(10)
b_comp_x = b_position^2 ⊗ b_fock
b_comp_p = b_momentum^2 ⊗ b_fock

Txp = transform(b_comp_x, b_comp_p)
nothing # hide
```

The function will then automatically identify the first to bases of the composite basis `b_comp_x` as position basis and define a corresponding FFT from `b_comp_p` to `b_comp_x`. Note, that it is also possible to specify the indexes of the bases one wants to transform. The above [`FFTOperator`](@ref QuantumOpticsBase.FFTOperator) is equivalent to writing `transform(b_comp_x, b_comp_p; index=[1, 2])`. This can be used if one does not want to FFT all position/momentum bases of a composite basis. Additionally, in order to save memory one can specify the option argument `ket_only`, for example `transform(b_comp_x, b_comp_p; ket_only=true)`. The resulting [`FFTOperator`](@ref QuantumOpticsBase.FFTOperator) is then of the subtype [`FFTKets`](@ref QuantumOpticsBase.FFTKets) and can only be applied to [`Ket`](@ref) states. This means, that this type of [`FFTOperator`](@ref QuantumOpticsBase.FFTOperator) can only be used when calculating the time evolution according to a Schrödinger equation, as opposed to the more general [`FFTOperators`](@ref QuantumOpticsBase.FFTOperators), which can also handle operators. However, when treating large systems, the memory efficiency is much better.

## [Additional functions](@id particle: Additional functions)

* [`spacing`](@ref)
* [`samplepoints`](@ref)


## [Examples](@id particle: Examples)

* [Particle in harmonic trap potential](@ref)
* [Gaussian wave packet running into a potential barrier](@ref)
* [N-Particles in a double-well potential](@ref)
* [Dynamics of a two-dimensional wavepacket hitting a Gaussian potential](@ref)
