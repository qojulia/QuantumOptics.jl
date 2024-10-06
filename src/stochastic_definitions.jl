"""
    stochastic.homodyne_carmichael(H0, C, theta)

Helper function that defines the functions needed to compute homodyne detection
trajectories according to Carmichael with `stochastic.schroedinger_dynamic`.

# Arguments
* `H0`: The deterministic, time-independent system Hamiltonian.
* `C`: Collapse operator (or vector of operators) of the detected output channel(s).
* `theta`: The phase difference between the local oscillator and the signal field.
    Defines the operator of the measured quadrature as
    ``X_\\theta = C e^{-i\\theta} + C^\\dagger e^{i\\theta}``. Needs to be a
    vector of the same length as `C` if `C` is a vector.
* `normalize_expect=true`: Specify whether or not to normalize the state vector
    when the expectation value in the nonlinear term is calculated. NOTE:
    should only be set to `false` if the state is guaranteed to be normalized,
    e.g. by setting `normalize_state=true` in `stochastic.schroedinger_dynamic`.

Returns `(fdeterm, fstoch)`, where `fdeterm(t, psi) -> H` and
`fstoch(t, psi) -> Hs` are functions returning the deterministic and stochastic
part of the Hamiltonian required for calling `stochastic.schroedinger_dynamic`.

The deterministic and stochastic parts of the Hamiltonian are constructed as

```math
H_{det} = H_0 + H_{nl},
```

where

```math
H_{nl} = iCe^{-i\\theta} \\langle X_\\theta \\rangle - \\frac{i}{2} C^\\dagger C,
```

and

```math
H_s = iCe^{-i\\theta}.
```
"""
function homodyne_carmichael(H0::AbstractOperator, C::Vector{T}, theta::Vector{R};
            normalize_expect::Bool=true) where {T <: AbstractOperator, R <: Real}
    n = length(C)
    @assert n == length(theta)
    Hs = 1.0im*C .* exp.(-1.0im .* theta)
    X = C .* exp.(-1.0im .* theta) + dagger.(C) .* exp.(1.0im .* theta)
    CdagC = -0.5im .* dagger.(C) .* C

    fstoch(t, psi::StateVector) = Hs
    if normalize_expect
        function H_nl_n(psi::StateVector)
            psi_n = normalize(psi)
            sum(expect(X[i], psi_n)*Hs[i] + CdagC[i] for i=1:n)
        end
        fdeterm_n(t, psi::StateVector) = H0 + H_nl_n(psi)
        return fdeterm_n, fstoch
    else
        H_nl_un(psi::StateVector) =
            sum(expect(X[i], psi)*Hs[i] + CdagC[i] for i=1:n)
        fdeterm_un(t, psi::StateVector) = H0 + H_nl_un(psi)
        return fdeterm_un, fstoch
    end
end
homodyne_carmichael(H0::AbstractOperator, C::AbstractOperator, theta::Real; kwargs...) =
    homodyne_carmichael(H0, [C], [theta]; kwargs...)
