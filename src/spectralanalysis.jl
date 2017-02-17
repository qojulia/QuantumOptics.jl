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
Simultaneously diagonalize two commuting Hermitian operators A and B.

This is done by diagonalizing a random linear combination of the operators.
The eigenvalues are computed by :math:`a = \\langle \\psi |A|\\psi\\rangle` and
it is checked whether the eigenvectors fulfill the equation
:math:`A|\\psi\\rangle = a|\\psi\\rangle`.

Arguments
---------

A
  Sparse or dense operator.
B
  Sparse or dense operator.

Keyword arguments
-----------------

sortby (optional)
  Integer that is either 1 or 2, specifying if the resulting common eigenvectors
  should be sorted by the eigenvalues of A (1) or B (2) in increasing order.
  Default is 1.

atol (optional)
  kwarg of Base.isapprox specifying the tolerance of the approximate check
  Default is 1e-14.

rtol (optional)
  kwarg of Base.isapprox specifying the tolerance of the approximate check
  Default is 1e-14.
"""

function simdiag(A::DenseOperator, B::DenseOperator; sortby::Int=1, atol::Real=1e-14, rtol::Real=1e-14)

  # Check input
  A == dagger(A) && B == dagger(B) ? nothing : error("Non-hermitian operator given!")
  sortby == 1 || sortby == 2 ? nothing : error("Require sortby::Int = 1, 2!")

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
    checkA = isapprox(A.data*v[:, i] - dA[i]*v[:, i], zeros(length(d)); atol=atol, rtol=rtol)
    checkB = isapprox(B.data*v[:, i] - dB[i]*v[:, i], zeros(length(d)); atol=atol, rtol=rtol)
    checkA && checkB ? nothing : error("Simultaneous diagonalization failed!")
  end

  index = sortby == 1 ? sortperm(real(dA)) : sortperm(real(dB))
  real(dA[index]), real(dB[index]), v[:, index]
end

simdiag(A::SparseOperator, B::SparseOperator; kwargs...) = simdiag(full(A), full(B); kwargs...)

end # module
