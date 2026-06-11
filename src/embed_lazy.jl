using QuantumOpticsBase

"""
    embed_lazy(basis, index, op)
    embed_lazy(basis_l, basis_r, index, op)
    embed_lazy(basis, indices, operators)

Embed `op` into a composite `basis` at position `index` (or positions `indices`)
without materialising the full tensor-product matrix. Returns a `LazyTensor` or
`LazySum` that defers allocation until you call `dense` or `sparse`.

On a non-composite `Basis` the operator is returned as-is after a compatibility check.

See also: [`embed`](@ref)
"""
function embed_lazy end


"""
    _check_subsystem(basis_l, basis_r, index, op)

Verify that `op` is compatible with the subsystem at `index` in `basis_l` and
`basis_r`. Throws `IncompatibleBases` if the bases do not match.
"""
function _check_subsystem(basis_l, basis_r, index, op)
    (basis_l.bases[index] == op.basis_l && basis_r.bases[index] == op.basis_r) ||
        throw(IncompatibleBases())
end

"""
    _check_subsystems(basis_l, basis_r, indices, ops)

Verify that each operator in `ops` is compatible with the corresponding subsystem
in `basis_l` and `basis_r`. Also checks that `indices` and `ops` have the same
length before iterating.
"""
function _check_subsystems(basis_l, basis_r, indices, ops)
    length(indices) == length(ops) ||
        throw(ArgumentError("length(indices) must equal length(operators)"))
    for (i, op) in zip(indices, ops)
        _check_subsystem(basis_l, basis_r, i, op)
    end
end


"""
    embed_lazy(basis_l, basis_r, index, op)
    embed_lazy(basis_l, basis_r, index, op::LazyTensor)
    embed_lazy(basis_l, basis_r, index, op::LazySum)
    embed_lazy(basis_l, basis_r, index, op::TimeDependentSum)

Single-index embedding into a `CompositeBasis`. The asymmetric `(basis_l, basis_r)`
form. The symmetric shortcut `embed_lazy(b, index, op)` at the end delegates here 
so there is only one place to update if the logic changes.

Each operator type dispatches separately because the wrapping differs: plain operators
go into a `LazyTensor` directly, `LazySum` maps recursively over its terms, and
`TimeDependentSum` re-embeds only the static part while preserving the time-varying
coefficients unchanged.
"""
function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    index::Integer, op::AbstractOperator)
    _check_subsystem(bl, br, index, op)
    LazyTensor(bl, br, index, op)
end

function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    index::Integer, op::LazyTensor)
    length(op.operators) == 1 ||
        throw(ArgumentError("single-index embed_lazy needs a single-operator LazyTensor"))
    LazyTensor(bl, br, index, first(values(op.operators)), op.factor)
end

function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    index::Integer, op::LazySum)
    LazySum(bl, br, op.factors, map(o -> embed_lazy(bl, br, index, o), op.operators))
end

function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    index::Integer, op::TimeDependentSum)
    TimeDependentSum(
        QuantumOpticsBase.coefficients(op),
        embed_lazy(bl, br, index, QuantumOpticsBase.static_operator(op));
        init_time = current_time(op)
    )
end

embed_lazy(b::CompositeBasis, index::Integer, op) = embed_lazy(b, b, index, op)


"""
    embed_lazy(basis_l, basis_r, indices, op::LazyTensor)
    embed_lazy(basis_l, basis_r, indices, op::LazySum)
    embed_lazy(basis_l, basis_r, indices, operators)

Multi-index embedding into a `CompositeBasis`. A `LazyTensor` already carries its
own operator dictionary, so we validate the existing operators against the new basis
and re-wrap rather than constructing from scratch. A plain vector of operators is
assembled into a fresh `LazyTensor`. `LazySum` recurses over its terms the same way
as in the single-index case.
"""
function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    indices, op::LazyTensor)
    _check_subsystems(bl, br, indices, op.operators)
    LazyTensor(bl, br, indices, op.operators, op.factor)
end

function embed_lazy(bl::CompositeBasis, br::CompositeBasis,
                    indices, op::LazySum)
    LazySum(bl, br, op.factors, map(o -> embed_lazy(bl, br, indices, o), op.operators))
end

function embed_lazy(bl::CompositeBasis, br::CompositeBasis, indices,
                    operators::AbstractVector{<:AbstractOperator})
    _check_subsystems(bl, br, indices, operators)
    LazyTensor(bl, br, indices, Tuple(operators))
end

embed_lazy(b::CompositeBasis, indices, op) = embed_lazy(b, b, indices, op)


"""
    embed_lazy(b::Basis, index, op)

When `b` is not a `CompositeBasis` there is nothing to embed into. The operator already 
lives on the full space. We verify that `index` is 1 (the only meaningful position 
on a non-composite basis) and return `op` unchanged.
"""
function embed_lazy(b::Basis, index::Integer, op)
    index == 1 || throw(ArgumentError("index must be 1 for a non-composite basis"))
    op
end
