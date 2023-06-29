using Arpack
import KrylovKit: eigsolve

const nonhermitian_warning = "The given operator is not hermitian. If this is due to a numerical error make the operator hermitian first by calculating (x+dagger(x))/2 first."

mutable struct DiagStrategy{DT}
    n::Int
    v0::Union{Nothing, AbstractVector}
end
DiagStrategy{T}(n::Int) where T = DiagStrategy{T}(n, nothing)
const LapackDiag = DiagStrategy{:lapack}
const ArpackDiag = DiagStrategy{:arpack}
const KrylovDiag = DiagStrategy{:krylov}
export LapackDiag, ArpackDiag, KrylovDiag

arithmetic_unary_error = QuantumOpticsBase.arithmetic_unary_error
DiagStrategy(op::AbstractOperator) = arithmetic_unary_error("DiagStrategy", op)
function DiagStrategy(op::DataOperator)
    basis(op) # Checks basis match
    DiagStrategy(op.data)
end
DiagStrategy(m::Matrix) = LapackDiag(size(m)[1], nothing)
DiagStrategy(m::SparseMatrixCSC) = ArpackDiag(6, rand(eltype(m), size(m)[1]))
DiagStrategy(m::AbstractMatrix) = ArgumentError("Cannot detect DiagStrategy for array type $(typeof(m))")
function _assert_starting_vector(ds::DiagStrategy)
    @assert ds.v0 !== nothing "Starting vector required for $(typeof(ds)) strategy"
end

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
function eigenstates(op::Operator, ds::LapackDiag; warning=true)
    b = basis(op)
    if ishermitian(op)
        D, V = eigen(Hermitian(op.data), 1:ds.n)
        states = [Ket(b, V[:, k]) for k=1:length(D)]
        return D, states
    else
        warning && @warn(nonhermitian_warning)
        D, V = eigen(op.data)
        states = [Ket(b, V[:, k]) for k=1:length(D)]
        perm = sortperm(D, by=real)
        permute!(D, perm)
        permute!(states, perm)
        return D[1:ds.n], states[1:ds.n]
    end
end

"""
For sparse operators by default it only returns the 6 lowest eigenvalues.
"""
function eigenstates(op::Operator, ds::ArpackDiag; warning::Bool=true,
        info::Bool=true, kwargs...)
    b = basis(op)
    # TODO: Change to sparse-Hermitian specific algorithm if more efficient
    ishermitian(op) || (warning && @warn(nonhermitian_warning))
    info && println("INFO: Defaulting to sparse diagonalization.
        If storing the full operator is possible, it might be faster to do
        eigenstates(dense(op)). Set info=false to turn off this message.")
    if ds.v0 === nothing
        D, V = eigs(op.data; which=:SR, nev=ds.n, kwargs...)
    else
        D, V = eigs(op.data; which=:SR, nev=ds.n, v0=ds.v0, kwargs...)
    end
    states = [Ket(b, V[:, k]) for k=1:length(D)]
    D, states
end

function eigenstates(op::Operator, ds::KrylovDiag; warning::Bool=true,
    info::Bool=true, kwargs...)
    b = basis(op)
    ishermitian(op) || (warning && @warn(nonhermitian_warning))
    info && println("INFO: Defaulting to sparse diagonalization.
        If storing the full operator is possible, it might be faster to do
        eigenstates(dense(op)). Set info=false to turn off this message.")
    _assert_starting_vector(ds)
    if ds.v0 === nothing
        D, Vs = eigsolve(op.data, ds.n, :SR; kwargs...)
    else
        D, Vs = eigsolve(op.data, ds.v0, ds.n, :SR; kwargs...)
    end
    states = [Ket(b, v) for v=Vs]
    D, states
end

function eigenstates(op::AbstractOperator, n::Union{Int,Nothing}=nothing; warning=true, kw...)
    ds = DiagStrategy(op)
    n !== nothing && (ds.n = n)
    eigenstates(op, ds; warning=warning, kw...)
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
function eigenenergies(op::Operator, ds::LapackDiag; warning=true)
    if ishermitian(op)
        D = eigvals(Hermitian(op.data), 1:ds.n)
        return D
    else
        warning && @warn(nonhermitian_warning)
        D = eigvals(op.data)
        sort!(D, by=real)
        return D[1:ds.n]
    end
end

"""
For sparse operators by default it only returns the 6 lowest eigenvalues.
"""
eigenenergies(op::Operator, ds::ArpackDiag; kwargs...) = eigenstates(op, ds; kwargs...)[1]

function eigenenergies(op::AbstractOperator, n::Union{Int,Nothing}=nothing; kw...)
    ds = DiagStrategy(op)
    n !== nothing && (ds.n = n)
    eigenenergies(op, ds; kw...)
end

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
function simdiag(ops::Vector{T}; atol::Real=1e-14, rtol::Real=1e-14) where T<:DenseOpType
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
