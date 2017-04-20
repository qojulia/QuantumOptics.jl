module spectralanalysis

import Base.eig, Base.eigs, Base.eigvals, Base.eigvals!
using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export simdiag


"""
    eig(op::DenseOperator, args...)

Diagonalize a dense operator. This is just a thin wrapper around julia's
`eig` function. More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/]

# Returns
* `D`: Vector of eigenvalues sorted from smalles (abs) to largest.
* `states`: Vector of Kets in the basis of A sorted by D.
"""
function eig(A::DenseOperator, args...)
  check_samebases(A)
  b = A.basis_l
  if ishermitian(A)
    D, V = eig(Hermitian(A.data), args...)
    states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
  else
    D, V = eig(A.data, args...)
    states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
    perm = sortperm(D, by=abs)
    permute!(D, perm)
    permute!(states, perm)
  end
  return D, states
end

"""
    eigs(op::SparseOperator, args...)

Diagonalize a sparse operator. This is just a thin wrapper around julia's
`eigs` function. More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/]

# Returns
* `D`: Vector of eigenvalues sorted from smalles (abs) to largest.
* `states`: Vector of Kets in the basis of A sorted by D.
"""
function eigs(A::SparseOperator, args...; kwargs...)
    check_samebases(A)
    b = A.basis_l
    if ishermitian(A)
        D, V = eigs(Hermitian(A.data), args...; kwargs...)
        states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
    else
        D, V = eigs(A.data, args...; kwargs...)
        states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
        perm = sortperm(D, by=abs)
        permute!(D, perm)
        permute!(states, perm)
    end
    return D, states
end

arithmetic_unary_error = operators.arithmetic_unary_error
eig(A::Operator, args...) = arithmetic_unary_error(eig, A)
eigs(A::Operator, args...) = arithmetic_unary_error(eig, A)


"""
    eigvals(op::DenseOperator, args...)

Compute eigenvalues of an operator. This is just a thin wrapper around julia's
`eigvals` function. More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/]

# Returns
* `evals_sorted`: Vector containing eigenvalues sorted from smallest to largest
        absolute value.
"""
eigvals(A::DenseOperator, args...) = ishermitian(A) ? eigvals(Hermitian(A.data), args...) : sort(eigvals(A.data), by=abs)

"""
    eigvals!(op::DenseOperator, args...)

Compute eigenvalues of an operator. This is just a thin wrapper around julia's
`eigvals` function. More details can be found at
[http://docs.julialang.org/en/stable/stdlib/linalg/]

# Returns
* `evals_sorted`: Vector containing eigenvalues sorted from smallest to largest
        absolute value.
"""
eigvals!(A::DenseOperator, args...) = ishermitian(A) ? eigvals!(Hermitian(A.data), args...) : sort(eigvals!(A.data), by=abs)

eigvals(A::Operator, args...) = arithmetic_unary_error(eigvals, A)
eigvals!(A::Operator, args...) = arithmetic_unary_error(eigvals!, A)


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
function simdiag{T<:DenseOperator}(ops::Vector{T}; atol::Real=1e-14, rtol::Real=1e-14)
    # Check input
    for A=ops
        if !ishermitian(A)
            error("Non-hermitian operator given!")
        end
    end

    d, v = eig(sum(ops).data)

    evals = [Vector{Complex128}(length(d)) for i=1:length(ops)]
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
