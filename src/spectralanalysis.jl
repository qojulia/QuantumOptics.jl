using Arpack
import KrylovKit: eigsolve

const nonhermitian_warning = "The given operator is not hermitian. If this is due to a numerical error make the operator hermitian first by calculating (x+dagger(x))/2 first."

"""
    abstract type DiagStrategy

Represents an algorithm used to find eigenvalues and eigenvectors of some operator.
Subtypes of this abstract type correspond to concrete routines. See `LapackDiag`,
`KrylovDiag` for more info.
"""
abstract type DiagStrategy end

"""
    LapackDiag <: DiagStrategy

Represents the `LinearArgebra.eigen` diagonalization routine.
The only parameter `n` represents the number of (lowest) eigenvectors.
"""
struct LapackDiag <: DiagStrategy
    n::Int
end

"""
    KrylovDiag <: DiagStrategy

Represents the `KrylovKit.eigsolve` routine. Implements the Lanczos & Arnoldi algorithms.
"""
struct KrylovDiag{VT} <: DiagStrategy
    n::Int
    v0::VT
    krylovdim::Int
end
"""
    KrylovDiag(n::Int, [v0=nothing, krylovdim::Int=n + 30])

Parameters:
- `n`: The number of eigenvectors to find
- `v0`: The starting vector. By default it is `nothing`, which means it will be a random dense
`Vector`. This will not work for non-trivial array types like from `CUDA.jl`, so you might want
to define a new method for the `QuantumOptics.get_starting_vector` function.
- `krylovdim`: The upper bound for dimenstion count of the emerging Krylov space.
"""
KrylovDiag(n::Int, v0=nothing) = KrylovDiag(n, v0, n + 30)
Base.print(io::IO, kds::KrylovDiag) =
    print(io, "KrylovDiag($(kds.n))")

arithmetic_unary_error = QuantumOpticsBase.arithmetic_unary_error
"""
    detect_diagstrategy(op::Operator; kw...)

Find a `DiagStrategy` for the given operator; processes the `kw` keyword arguments
and automatically sets parameters of the resulting `DiagStrategy`object.
Returns a tuple of the `DiagStrategy` and unprocessed keyword arguments from `kw`.
"""
function detect_diagstrategy(op::DataOperator; kw...)
    QuantumOpticsBase.check_samebases(op)
    detect_diagstrategy(op.data; kw...)
end
detect_diagstrategy(op::AbstractOperator; kw...) = arithmetic_unary_error("detect_diagstrategy", op)

"""
    get_starting_vector(m::AbstractMatrix)

Generate a default starting vector for Arnoldi-like iterative methods for matrix `m`.
"""
get_starting_vector(::SparseMatrixCSC) = nothing
function detect_diagstrategy(m::AbstractSparseMatrix; kw...)
    if get(kw, :info, true)
        @info "Defaulting to sparse diagonalization for sparse operator. If storing the full operator is possible, it might be faster to do `eigenstates(dense(op))`. Set `info=false` to turn off this message."
    end
    nev = get(kw, :n, 6)
    v0 = get(kw, :v0, get_starting_vector(m))
    krylovdim = get(kw, :krylovdim, nev + 30)
    new_kw = Base.structdiff(values(kw), NamedTuple{(:n, :v0, :krylovdim, :info)})
    return KrylovDiag(nev, v0, krylovdim), new_kw
end
function detect_diagstrategy(m::Matrix; kw...)
    nev = get(kw, :n, size(m)[1])
    new_kw = Base.structdiff(values(kw), NamedTuple{(:n, :info)})
    return LapackDiag(nev), new_kw
end
"""
    detect_diagstrategy(m::AbstractMatrix; kw...)

Same as above, but dispatches on different internal array types.
"""
detect_diagstrategy(m::T; _...) where T<:AbstractMatrix = throw(ArgumentError(
    """Cannot detect DiagStrategy for array type $(typeof(m)).
    Consider defining `QuantumOptics.detect_diagstrategy(::$T; kw...)` method.
    Refer to `QuantumOptics.detect_diagstrategy` docstring for more info."""))

"""
    eigenstates(op::Operator[, n::Int; warning=true, kw...])

Calculate the lowest n eigenvalues and their corresponding eigenstates. By default `n` is
equal to the matrix size for dense matrices; for sparse matrices the default value is 6.

This is just a thin wrapper around julia's `LinearArgebra.eigen` and `KrylovKit.eigsolve`
functions. Which of them is used depends on the type of the given operator. If more control
about the way the calculation is done is needed, use the method instance with `DiagStrategy`
(see below).

NOTE: Especially for small systems full diagonalization with Julia's `eigen`
function is often more desirable. You can convert a sparse operator `A` to a
dense one using `dense(A)`.

If the given operator is non-hermitian a warning is given. This behavior
can be turned off using the keyword `warning=false`.

## Optional arguments
- `n`: It can be a keyword argument too!
- `v0`: The starting vector for Arnoldi-like iterative methods.
- `krylovdim`: The upper bound for dimenstion count of the emerging Krylov space.
"""
function eigenstates(op::AbstractOperator; kw...)
    ds, kwargs_rem = detect_diagstrategy(op; kw...)
    eigenstates(op, ds; kwargs_rem...)
end
eigenstates(op::AbstractOperator, n::Int; warning=true, kw...) =
    eigenstates(op; warning=warning, kw..., n=n)

"""
    eigenstates(op::Operator, ds::DiagStrategy[; warning=true, kw...])

Calculate the lowest eigenvalues and their corresponding eigenstates of the `op` operator
using the `ds` diagonalization strategy. The `kw...` arguments can be passed to the exact
function that does the diagonalization (like `KrylovKit.eigsolve`).
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

function eigenstates(op::Operator, ds::KrylovDiag; warning::Bool=true, kwargs...)
    b = basis(op)
    ishermitian(op) || (warning && @warn(nonhermitian_warning))
    if ds.v0 === nothing
        D, Vs = eigsolve(op.data, ds.n, :SR; krylovdim = ds.krylovdim, kwargs...)
    else
        D, Vs = eigsolve(op.data, ds.v0, ds.n, :SR; krylovdim = ds.krylovdim, kwargs...)
    end
    states = [Ket(b, Vs[k]) for k=1:ds.n]
    D[1:ds.n], states
end

"""
    eigenenergies(op::AbstractOperator[, n::Int; warning=true, kwargs...])

Calculate the lowest n eigenvalues of given operator.

If the given operator is non-hermitian a warning is given. This behavior
can be turned off using the keyword `warning=false`.

See `eigenstates` for more info.
"""
function eigenenergies(op::AbstractOperator; kw...)
    ds, kw_rem = detect_diagstrategy(op; kw...)
    eigenenergies(op, ds; kw_rem...)
end
eigenenergies(op::AbstractOperator, n::Int; kw...) = eigenenergies(op; kw..., n=n)

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

# Call eigenstates
eigenenergies(op::Operator, ds::DiagStrategy; kwargs...) = eigenstates(op, ds; kwargs...)[1]

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
