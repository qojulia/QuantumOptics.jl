module metrics

export tracenorm, tracenorm_h, tracenorm_nh,
        tracedistance, tracedistance_h, tracedistance_nh,
        entropy_vn, fidelity

using ..bases, ..operators, ..operators_dense

"""
    tracenorm(rho)

Trace norm of `rho`.

It is defined as

```math
T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\}.
```

Depending if `rho` is hermitian either [`tracenorm_h`](@ref) or
[`tracenorm_nh`](@ref) is called.
"""
function tracenorm(rho::DenseOperator)
    check_samebases(rho)
    ishermitian(rho) ? tracenorm_h(rho) : tracenorm_nh(rho)
end
function tracenorm{T<:Operator}(rho::T)
    throw(ArgumentError("tracenorm not implemented for $(T). Use dense operators instead."))
end

"""
    tracenorm_h(rho)

Trace norm of `rho`.

It uses the identity

```math
T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\} = \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of `rho`.
"""
function tracenorm_h(rho::DenseOperator)
    check_samebases(rho)
    s = eigvals(Hermitian(rho.data))
    sum(abs.(s))
end
function tracenorm_h{T<:Operator}(rho::T)
    throw(ArgumentError("tracenorm_h not implemented for $(T). Use dense operators instead."))
end


"""
    tracenorm_nh(rho)

Trace norm of `rho`.

Note that in this case `rho` doesn't have to be represented by a square
matrix (i.e. it can have different left-hand and right-hand bases).

It uses the identity

```math
    T(ρ) = Tr\\{\\sqrt{ρ^† ρ}\\} = \\sum_i σ_i
```

where ``σ_i`` are the singular values of `rho`.
"""
tracenorm_nh(rho::DenseOperator) = sum(svdvals(rho.data))
function tracenorm_nh{T<:Operator}(rho::T)
    throw(ArgumentError("tracenorm_nh not implemented for $(T). Use dense operators instead."))
end


"""
    tracedistance(rho, sigma)

Trace distance between `rho` and `sigma`.

It is defined as

```math
T(ρ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ}}\\}.
```

It calls [`tracenorm`](@ref) which in turn either uses [`tracenorm_h`](@ref)
or [`tracenorm_nh`](@ref) depending if ``ρ-σ`` is hermitian or not.
"""
tracedistance(rho::DenseOperator, sigma::DenseOperator) = 0.5*tracenorm(rho - sigma)
function tracedistance{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance not implemented for $(T). Use dense operators instead."))
end

"""
    tracedistance_h(rho, sigma)

Trace distance between `rho` and `sigma`.

It uses the identity

```math
T(ρ) = \\frac{1}{2} Tr\\{\\sqrt{ρ^† ρ}\\} = \\frac{1}{2} \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of `rho`.
"""
tracedistance_h(rho::DenseOperator, sigma::DenseOperator) = 0.5*tracenorm_h(rho - sigma)
function tracedistance_h{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance_h not implemented for $(T). Use dense operators instead."))
end

"""
    tracedistance_nh(rho, sigma)

Trace distance between `rho` and `sigma`.

Note that in this case `rho` and `sigma` don't have to be represented by square
matrices (i.e. they can have different left-hand and right-hand bases).

It uses the identity

```math
    T(ρ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\}
         = \\frac{1}{2} \\sum_i σ_i
```

where ``σ_i`` are the singular values of `rho` - `sigma`.
"""
tracedistance_nh(rho::DenseOperator, sigma::DenseOperator) = 0.5*tracenorm_nh(rho - sigma)
function tracedistance_nh{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance_nh not implemented for $(T). Use dense operators instead."))
end


"""
    entropy_vn(rho)

Von Neumann entropy of a density matrix.

The Von Neumann entropy of a density operator is defined as

```math
S(ρ) = -Tr(ρ \\log(ρ)) = -\\sum_n λ_n\\log(λ_n)
```

where ``λ_n`` are the eigenvalues of the density matrix ``ρ``, ``\\log`` is the
natural logarithm and ``\\log(0) ≡ 0``.
"""
entropy_vn(rho::DenseOperator) = sum([d == 0 ? 0 : -d*log(d) for d=eigvals(rho.data)])

"""
    fidelity(rho, sigma)

Fidelity of two density operators.

The fidelity of two density operators ``\ρ`` and ``σ`` is defined by

```math
F(ρ, σ) = Tr\\left(\\sqrt{\\sqrt{ρ}σ\\sqrt{ρ}}\\right),
```

where ``\\sqrt{ρ}=\\sum_n\\sqrt{λ_n}|ψ⟩⟨ψ|``.
"""
fidelity(rho::DenseOperator, sigma::DenseOperator) = trace(sqrtm(sqrtm(rho.data)*sigma.data*sqrtm(rho.data)))

end # module
