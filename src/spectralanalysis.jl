module spectralanalysis

export eigenstates, eigenenergies, simdiag

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse
using Arpack, LinearAlgebra, SparseArrays


const nonhermitian_warning = "The given operator is not hermitian. If this is due to a numerical error make the operator hermitian first by calculating (x+dagger(x))/2 first."

"""
    eigenstates(op::AbstractOperator[, n::Int; warning=true])

Calculate the lowest n eigenvalues and their corresponding eigenstates.

This is just a thin wrapper around julia's `eigen` and `eigs` functions. Which
of them is used depends on the type of the given operator. If more control
about the way the calculation is done is needed, use the functions directly.
More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/].

NOTE: Especially for small systems full diagonalization with Julia's `eigen`
function is often more desirable. You can convert a sparse operator `A` to a
dense one using `dense(A)`.

If the given operator is non-hermitian a warning is given. This behavior
can be turned off using the keyword `warning=false`.
"""
function eigenstates(op::Operator{B,B,T}, n::Int=length(basis(op)); warning=true) where {B<:Basis,T<:Matrix{ComplexF64}}
    b = basis(op)
    if ishermitian(op)
        D, V = eigen(Hermitian(op.data), 1:n)
        states = [Ket(b, V[:, k]) for k=1:length(D)]
        return D, states
    else
        warning && warn(nonhermitian_warning)
        D, V = eigen(op.data)
        states = [Ket(b, V[:, k]) for k=1:length(D)]
        perm = sortperm(D, by=real)
        permute!(D, perm)
        permute!(states, perm)
        return D[1:n], states[1:n]
    end
end

"""
For sparse operators by default it only returns the 6 lowest eigenvalues.
"""
function eigenstates(op::Operator{B,B,T}, n::Int=6; warning::Bool=true,
        info::Bool=true, kwargs...) where {B<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}}
    b = basis(op)
    # TODO: Change to sparse-Hermitian specific algorithm if more efficient
    ishermitian(op) || (warning && warn(nonhermitian_warning))
    info && println("INFO: Defaulting to sparse diagonalization.
        If storing the full operator is possible, it might be faster to do
        eigenstates(dense(op)). Set info=false to turn off this message.")
    D, V = eigs(op.data; which=:SR, nev=n, kwargs...)
    states = [Ket(b, V[:, k]) for k=1:length(D)]
    D, states
end


"""
    eigenenergies(op::AbstractOperator[, n::Int; warning=true])

Calculate the lowest n eigenvalues.

This is just a thin wrapper around julia's `eigvals`. If more control
about the way the calculation is done is needed, use the function directly.
More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/].

If the given operator is non-hermitian a warning is given. This behavior
can be turned off using the keyword `warning=false`.
"""
function eigenenergies(op::Operator{B,B,T}, n::Int=length(basis(op)); warning=true) where {B<:Basis,T<:Matrix{ComplexF64}}
    b = basis(op)
    if ishermitian(op)
        D = eigvals(Hermitian(op.data), 1:n)
        return D
    else
        warning && warn(nonhermitian_warning)
        D = eigvals(op.data)
        sort!(D, by=real)
        return D[1:n]
    end
end

"""
For sparse operators by default it only returns the 6 lowest eigenvalues.
"""
eigenenergies(op::Operator{B,B,T}, n::Int=6; kwargs...) where {B<:Basis,T<:SparseMatrixCSC{ComplexF64,Int}} =
    eigenstates(op, n; kwargs...)[1]


arithmetic_unary_error = operators.arithmetic_unary_error
eigenstates(op::AbstractOperator, n::Int=0) = arithmetic_unary_error("eigenstates", op)
eigenenergies(op::AbstractOperator, n::Int=0) = arithmetic_unary_error("eigenenergies", op)


"""
    simdiag(ops; atol, rtol)

Simultaneously diagonalize commuting Hermitian operators specified in `ops`.

This is done by diagonalizing the sum of the operators. The eigenvalues are
computed by ``a = ⟨ψ|A|ψ⟩`` and it is checked whether the eigenvectors fulfill
the equation ``A|ψ⟩ = a|ψ⟩``.

# Arguments
* `ops`: Vector of sparse or dense operators.
* `atol=1e-14`: kwarg of Base.isapprox specifying the tolerance of the
        approximate check
* `rtol=1e-14`: kwarg of Base.isapprox specifying the tolerance of the
        approximate check

# Returns
* `evals_sorted`: Vector containing all vectors of the eigenvalues sorted
        by the eigenvalues of the first operator.
* `v`: Common eigenvectors.
"""
function simdiag(ops::Vector{T}; atol::Real=1e-14, rtol::Real=1e-14) where {B<:Basis,D<:Matrix{ComplexF64},T<:Operator{B,B,D}}
    # Check input
    for A=ops
        if !ishermitian(A)
            error("Non-hermitian operator given!")
        end
    end

    d, v = eigen(sum(ops).data)

    evals = [Vector{ComplexF64}(undef, length(d)) for i=1:length(ops)]
    for i=1:length(ops), j=1:length(d)
        vec = ops[i].data*v[:, j]
        evals[i][j] = (v[:, j]'*vec)[1]
        if !isapprox(vec, evals[i][j]*v[:, j]; atol=atol, rtol=rtol)
            error("Simultaneous diagonalization failed!")
        end
    end

    index = sortperm(real(evals[1][:]))
    evals_sorted = [real(evals[i][index]) for i=1:length(ops)]
    return evals_sorted, v[:, index]
end

end # module
