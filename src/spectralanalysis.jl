module spectralanalysis

using ..states, ..operators, ..operators_dense, ..operators_sparse

export operatorspectrum, operatorspectrum_hermitian, eigenstates, eigenstates_hermitian, groundstate, simdiag


"""
Calculate the spectrum of a Hermitian operator.

Arguments
---------

H
    Sparse or dense operator.

Keyword arguments
-----------------

Nmax (optional)
    Number of eigenvalues that should be calculated.
"""
function operatorspectrum_hermitian(H::DenseOperator; Nmax::Union{Int, Void}=nothing)
    h = Hermitian(H.data)
    return Nmax == nothing ? eigvals(h) : eigvals(h, 1:Nmax)
end

operatorspectrum_hermitian(H::SparseOperator; Nmax::Union{Int, Void}=nothing) = real(operatorspectrum(H; Nmax=Nmax))


"""
Calculate the spectrum of a not necessarily Hermitian operator.

If the operator is known to be Hermitian use
:func:`operatorspectrum_hermitian(::DenseOperator)` instead.

Arguments
---------

H
    Sparse or dense operator.

Keyword arguments
-----------------

Nmax (optional)
    Number of eigenvalues that should be calculated.
"""
function operatorspectrum(H::DenseOperator; Nmax::Union{Int, Void}=nothing)
    if ishermitian(H.data)
        return operatorspectrum_hermitian(H; Nmax=Nmax)
    end
    s = eigvals(H.data)
    return Nmax == nothing ? s : s[1:Nmax]
end

function operatorspectrum(H::SparseOperator; Nmax::Union{Int, Void}=nothing)
    if Nmax == nothing
        Nmax = size(H.data, 2) - 2
    end
    d, nconv, niter, nmult, resid = eigs(H.data; nev=Nmax, which=:SR, ritzvec=false)
    return d
end


"""
Calculate the eigenstates of a Hermitian operator.

Arguments
---------

H
    Sparse or dense operator.

Keyword arguments
-----------------

Nmax (optional)
    Number of eigenstates that should be calculated.
"""
function eigenstates_hermitian(H::DenseOperator; Nmax::Union{Int, Void}=nothing)
    # h = Hermitian(H.data) -- In Julia v0.5 function eigfact does not support Hermitians
    h = H.data
    M = Nmax == nothing ? eigvecs(h) : eigvecs(h, 1:Nmax)
    b = Ket[]
    for k=1:size(M,2)
        push!(b, Ket(H.basis_r, M[:,k]))
    end
    return b
end


"""
Calculate the eigenstates of a not necessarily Hermitian operator.

If the operator is known to be Hermitian use
:func:`eigenstates_hermitian(::DenseOperator)` instead.

Arguments
---------

H
    Sparse or dense operator.

Keyword arguments
-----------------

Nmax (optional)
    Number of eigenstates that should be calculated.
"""
function eigenstates(H::DenseOperator; Nmax::Union{Int, Void}=nothing)
    if ishermitian(H.data)
        return eigenstates_hermitian(H; Nmax=Nmax)
    end
    M = eigvecs(H.data)
    b = Ket[]
    for k=1:size(M,2)
        if Nmax!=nothing && k>Nmax
            break
        end
        push!(b, Ket(H.basis_r, M[:,k]))
    end
    return b
end

eigenstates_hermitian(H::SparseOperator; Nmax::Union{Int, Void}=nothing) = eigenstates(H; Nmax=Nmax)


function eigenstates(H::SparseOperator; Nmax::Union{Int, Void}=nothing)
    if Nmax == nothing
        Nmax = size(H.data, 2) - 2
    end
    d, M, nconv, niter, nmult, resid = eigs(H.data; nev=Nmax, which=:SR, ritzvec=true)
    b = Ket[]
    for k=1:size(M,2)
        push!(b, Ket(H.basis_r, M[:,k]))
    end
    return b
end


"""
Calculate the ground-state of a Hermitian operator.

This is just a shortcut for :func:`eigenstates_hermitian(H, Nmax=1)`

Arguments
---------

H
    Sparse or dense operator.

Keyword arguments
-----------------

Nmax (optional)
    Number of eigenstates that should be calculated.
"""
groundstate(H::Union{DenseOperator, SparseOperator}) = eigenstates_hermitian(H; Nmax=1)[1]


"""
Simultaneously diagonalize two commuting Hermitian operators.

This is done by diagonalizing a random linear combination of the operators
and checking if both operators alone are diagonalized by the resulting
eigenvectors.
"""

function simdiag(A::DenseOperator, B::DenseOperator; max_iter::Int=1, atol::Real=1e-15, rtol::Real=1e-15, sortby::Int=1)
  A != dagger(A) || B != dagger(B) ? error("Non-hermitian operator given!") : nothing

  sortby == 1 || sortby == 2 ? nothing : error("Require sortby::Int = 1, 2!")

  comm = A.data*B.data - B.data*A.data
  isapprox(comm, 0.0im*zeros(size(A.data)); atol=atol, rtol=rtol) ? nothing : error("Operators do not commute!")

  a, b = rand(2)
  while a == 0 || b == 0
    a, b = rand(2)
  end

  d, v = eig(a*A.data + b*B.data)

  dA = Vector{Complex128}(length(d))
  dB = Vector{Complex128}(length(d))
  for i=1:length(d)
    dA[i] = (v[:, i]'*A.data*v[:, i])[1]
    dB[i] = (v[:, i]'*B.data*v[:, i])[1]
    vA = A.data*v[:, i]
    vB = B.data*v[:, i]

    if dA[i] == 0
      checkA = isapprox(vA, zeros(length(d)); atol=atol, rtol=rtol)
    else
      checkA = isapprox((vA'*v[:, i])[1]/dA[i], 1.0; atol=atol, rtol=rtol)
    end

    if dB[i] == 0
      checkB = isapprox(vB, zeros(length(d)); atol=atol, rtol=rtol)
    else
      checkB = isapprox((vB'*v[:, i])[1]/dB[i], 1.0; atol=atol, rtol=rtol)
    end

    checkA && checkB ? nothing : error("Simultaneous diagonalization failed!")
  end

  index = sortby == 1 ? sortperm(real(dA)) : sortperm(real(dB))

  real(dA[index]), real(dB[index]), v[:, index]
end

simdiag(A::SparseOperator, B::SparseOperator; kwargs...) = simdiag(full(A), full(B); kwargs...)

end # module
