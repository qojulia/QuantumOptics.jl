module spectralanalysis

using ..states, ..operators, ..operators_dense, ..operators_sparse

export operatorspectrum, operatorspectrum_hermitian, eigenstates, eigenstates_hermitian, groundstate


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

end # module
