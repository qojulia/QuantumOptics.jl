module metrics

export tracedistance, tracedistance_general, tracenorm, tracenorm_general,
        entropy_vn, fidelity

using ..bases, ..operators, ..operators_dense


"""
    tracenorm(rho)

Trace norm of `rho`

It uses the identity

```math
T(ρ) = \\frac{1}{2} \\sum_i |λ_i|
```
"""
function tracenorm(rho::DenseOperator)
    check_samebases(rho)
    data = rho.data
    for i=1:size(data,1)
        data[i,i] = real(data[i,i])
    end
    s = eigvals(Hermitian(data))
    return 0.5*sum(abs.(s))
end

function tracenorm{T<:Operator}(rho::T)
    throw(ArgumentError("tracenorm not implemented for $(T). Use dense operators instead."))
end


"""
    tracenorm_general(rho)

Trace norm of `rho`.

It uses the identity

```math
    T(ρ) = \\frac{1}{2} Tr\\{\\sqrt{ρ^† ρ}\\}
```
"""
tracenorm_general(rho::DenseOperator) = 0.5*trace(sqrtm((dagger(rho)*rho).data))

function tracenorm_general{T<:Operator}(rho::T)
    throw(ArgumentError("tracenorm_general not implemented for $(T). Use dense operators instead."))
end



"""
    tracedistance(rho, sigma)

Trace distance between two density operators.

It uses the identity

```math
T(ρ, σ) = \\frac{1}{2} \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of the matrix ``ρ - σ``. This works only
if `rho` and `sigma` are density operators. For trace distances between
general operators use [`tracedistance_general`](@ref).
"""
tracedistance(rho::DenseOperator, sigma::DenseOperator) = tracenorm(rho - sigma)

function tracedistance{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance not implemented for $(T). Use dense operators instead."))
end


"""
    tracedistance_general(rho, sigma)

Trace distance between two operators.

It uses the identity

```math
    T(ρ, σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ-σ)^† (ρ-σ)}\\}
```
"""
tracedistance_general(rho::DenseOperator, sigma::DenseOperator) = tracenorm_general(rho - sigma)

function tracedistance_general{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance_general not implemented for $(T). Use dense operators instead."))
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
