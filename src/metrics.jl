module metrics

export tracenorm, tracenorm_h, tracenorm_nh,
        tracedistance, tracedistance_h, tracedistance_nh,
        entropy_vn, fidelity, ptranspose, PPT, negativity,
        logarithmic_negativity

using ..bases, ..operators, ..operators_dense, ..states
using LinearAlgebra

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
function tracenorm(rho::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}}
    ishermitian(rho) ? tracenorm_h(rho) : tracenorm_nh(rho)
end
function tracenorm(rho::T) where T<:AbstractOperator
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
function tracenorm_h(rho::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}}
    s = eigvals(Hermitian(rho.data))
    sum(abs.(s))
end
function tracenorm_h(rho::T) where T<:AbstractOperator
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
tracenorm_nh(rho::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}} = sum(svdvals(rho.data))
function tracenorm_nh(rho::T) where T<:AbstractOperator
    throw(ArgumentError("tracenorm_nh not implemented for $(T). Use dense operators instead."))
end


"""
    tracedistance(rho, sigma)

Trace distance between `rho` and `sigma`.

It is defined as

```math
T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\}.
```

It calls [`tracenorm`](@ref) which in turn either uses [`tracenorm_h`](@ref)
or [`tracenorm_nh`](@ref) depending if ``ρ-σ`` is hermitian or not.
"""
tracedistance(rho::Operator{B,B,T}, sigma::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}} = 0.5*tracenorm(rho - sigma)
function tracedistance(rho::T, sigma::T) where T<:AbstractOperator
    throw(ArgumentError("tracedistance not implemented for $(T). Use dense operators instead."))
end

"""
    tracedistance_h(rho, sigma)

Trace distance between `rho` and `sigma`.

It uses the identity

```math
T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\} = \\frac{1}{2} \\sum_i |λ_i|
```

where ``λ_i`` are the eigenvalues of `rho` - `sigma`.
"""
tracedistance_h(rho::Operator{B,B,T}, sigma::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}} = 0.5*tracenorm_h(rho - sigma)
function tracedistance_h(rho::T, sigma::T) where T<:AbstractOperator
    throw(ArgumentError("tracedistance_h not implemented for $(T). Use dense operators instead."))
end

"""
    tracedistance_nh(rho, sigma)

Trace distance between `rho` and `sigma`.

Note that in this case `rho` and `sigma` don't have to be represented by square
matrices (i.e. they can have different left-hand and right-hand bases).

It uses the identity

```math
    T(ρ,σ) = \\frac{1}{2} Tr\\{\\sqrt{(ρ - σ)^† (ρ - σ)}\\}
         = \\frac{1}{2} \\sum_i σ_i
```

where ``σ_i`` are the singular values of `rho` - `sigma`.
"""
tracedistance_nh(rho::Operator{B,B,T}, sigma::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}} = 0.5*tracenorm_nh(rho - sigma)
function tracedistance_nh(rho::T, sigma::T) where T<:AbstractOperator
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

# Arguments
* `rho`: Density operator of which to calculate Von Neumann entropy.
* `tol=1e-15`: Tolerance for rounding errors in the computed eigenvalues.
"""
function entropy_vn(rho::Operator{B,B,T}; tol::Float64=1e-15) where {B<:Basis,T<:Matrix{ComplexF64}}
    evals = eigvals(rho.data)
    evals[abs.(evals) .< tol] .= 0.0
    sum([d == 0 ? 0 : -d*log(d) for d=evals])
end
entropy_vn(psi::StateVector; kwargs...) = entropy_vn(dm(psi); kwargs...)

"""
    fidelity(rho, sigma)

Fidelity of two density operators.

The fidelity of two density operators ``ρ`` and ``σ`` is defined by

```math
F(ρ, σ) = Tr\\left(\\sqrt{\\sqrt{ρ}σ\\sqrt{ρ}}\\right),
```

where ``\\sqrt{ρ}=\\sum_n\\sqrt{λ_n}|ψ⟩⟨ψ|``.
"""
fidelity(rho::Operator{B,B}, sigma::Operator{B,B}) where B<:Basis = tr(sqrt(sqrt(rho.data)*sigma.data*sqrt(rho.data)))


"""
    ptranspose(rho, index)

Partial transpose of rho with respect to subspace specified by index.
"""
function ptranspose(rho::Operator{B,B}, index::Int=1) where B<:CompositeBasis

    # Define permutation
    N = length(rho.basis_l.bases)
    perm = [1:N;]
    perm[index] = N
    perm[N] = index

    # Permute indexed subsystem to last position
    rho_perm = permutesystems(rho, perm)

    # Transpose corresponding blocks
    m = Int(prod(rho_perm.basis_l.shape[1:N-1]))
    n = rho_perm.basis_l.shape[N]
    for i=1:n, j=1:n
        rho_perm.data[m*(i-1)+1:m*i, m*(j-1)+1:m*j] = permutedims(rho_perm.data[m*(i-1)+1:m*i, m*(j-1)+1:m*j])
    end

    return permutesystems(rho_perm, perm)
end


"""
    PPT(rho, index)

Peres-Horodecki criterion of partial transpose.
"""
PPT(rho::Operator{B,B,T}, index::Int) where {B<:Basis,T<:Matrix{ComplexF64}} = all(real.(eigvals(ptranspose(rho, index).data)) .>= 0.0)


"""
    negativity(rho, index)

Negativity of rho with respect to subsystem index.

The negativity of a density matrix ρ is defined as

```math
N(ρ) = \\|ρᵀ\\|,
```
where `ρᵀ` is the partial transpose.
"""
negativity(rho::Operator{B,B}, index::Int) where B<:CompositeBasis = 0.5*(tracenorm(ptranspose(rho, index)) - 1.0)


"""
    logarithmic_negativity(rho, index)

The logarithmic negativity of a density matrix ρ is defined as

```math
N(ρ) = \\log₂\\|ρᵀ\\|,
```
where `ρᵀ` is the partial transpose.
"""
logarithmic_negativity(rho::Operator{B,B}, index::Int) where B<:CompositeBasis = log(2, tracenorm(ptranspose(rho, index)))

end # module
