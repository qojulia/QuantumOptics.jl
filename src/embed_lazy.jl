# src/embed_lazy.jl
# Lazy embedding utilities for QuantumOptics.
#
# embed_lazy is defined here (not in QuantumOpticsBase, which does not provide
# this function) and is exported from the QuantumOptics module.

using QuantumOpticsBase

# ====================== Internal Basis Checks ======================

function _embed_lazy_check_basis(basis_l::CompositeBasis, basis_r::CompositeBasis,
                                 index::Integer, op::AbstractOperator)
    (basis_l.bases[index] == op.basis_l && basis_r.bases[index] == op.basis_r) ||
        throw(IncompatibleBases())
end

function _embed_lazy_check_basis(basis_l::CompositeBasis, basis_r::CompositeBasis,
                                 indices, operators)
    length(indices) == length(operators) ||
        throw(ArgumentError("embed_lazy requires length(indices) == length(operators)."))
    for (index, op) in zip(indices, operators)
        basis_l.bases[index] == op.basis_l || throw(IncompatibleBases())
        basis_r.bases[index] == op.basis_r || throw(IncompatibleBases())
    end
    return nothing
end

function _embed_lazy_check_basis(b::Basis, index::Integer, op::AbstractOperator)
    index == 1 || throw(ArgumentError("For non-composite basis, embed index must be 1."))
    b == op.basis_l || throw(IncompatibleBases())
    return nothing
end

function _embed_lazy_check_basis(b::Basis, index::Integer,
                                 ops::Union{Tuple,AbstractVector})
    index == 1 || throw(ArgumentError("For non-composite basis, embed index must be 1."))
    for op in ops
        b == op.basis_l || throw(IncompatibleBases())
    end
    return nothing
end

# ====================== embed_lazy ======================

"""
    embed_lazy(basis, index, op)
    embed_lazy(basis_l, basis_r, index, op)
    embed_lazy(basis, indices, operators)

Embed `op` into `basis` at position `index` (or positions `indices`) using
lazy operators (`LazyTensor`, `LazySum`). Unlike `embed`, this avoids
materialising the full tensor-product operator and therefore scales much better
for large composite systems.

For a non-composite `Basis` the operator is returned unchanged (after a
compatibility check), since no embedding is needed.
"""
function embed_lazy end

# --- AbstractOperator → LazyTensor ---

function embed_lazy(basis::CompositeBasis, index::Integer, op::AbstractOperator)
    _embed_lazy_check_basis(basis, basis, index, op)
    LazyTensor(basis, index, op)
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    index::Integer, op::AbstractOperator)
    _embed_lazy_check_basis(basis_l, basis_r, index, op)
    LazyTensor(basis_l, basis_r, index, op)
end

# --- LazyTensor (single-index shortcut) ---

function embed_lazy(basis::CompositeBasis, index::Integer, op::LazyTensor)
    length(op.operators) == 1 ||
        throw(ArgumentError("embed_lazy with a single index requires a single-operator LazyTensor."))
    LazyTensor(basis, index, first(values(op.operators)), op.factor)
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    index::Integer, op::LazyTensor)
    length(op.operators) == 1 ||
        throw(ArgumentError("embed_lazy with a single index requires a single-operator LazyTensor."))
    LazyTensor(basis_l, basis_r, index, first(values(op.operators)), op.factor)
end

# --- LazyTensor (multi-index) ---

function embed_lazy(basis::CompositeBasis, indices, op::LazyTensor)
    _embed_lazy_check_basis(basis, basis, indices, op.operators)
    LazyTensor(basis, basis, indices, op.operators, op.factor)
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    indices, op::LazyTensor)
    _embed_lazy_check_basis(basis_l, basis_r, indices, op.operators)
    LazyTensor(basis_l, basis_r, indices, op.operators, op.factor)
end

# --- LazySum ---

function embed_lazy(basis::CompositeBasis, index::Integer, op::LazySum)
    LazySum(basis, basis, op.factors,
            map(o -> embed_lazy(basis, index, o), op.operators))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    index::Integer, op::LazySum)
    LazySum(basis_l, basis_r, op.factors,
            map(o -> embed_lazy(basis_l, basis_r, index, o), op.operators))
end

function embed_lazy(basis::CompositeBasis, indices, op::LazySum)
    LazySum(basis, basis, op.factors,
            map(o -> embed_lazy(basis, indices, o), op.operators))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    indices, op::LazySum)
    LazySum(basis_l, basis_r, op.factors,
            map(o -> embed_lazy(basis_l, basis_r, indices, o), op.operators))
end

# --- TimeDependentSum ---

function embed_lazy(basis::CompositeBasis, index::Integer, op::TimeDependentSum)
    TimeDependentSum(QuantumOpticsBase.coefficients(op),
                     embed_lazy(basis, index, QuantumOpticsBase.static_operator(op));
                     init_time = current_time(op))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    index::Integer, op::TimeDependentSum)
    TimeDependentSum(QuantumOpticsBase.coefficients(op),
                     embed_lazy(basis_l, basis_r, index, QuantumOpticsBase.static_operator(op));
                     init_time = current_time(op))
end

function embed_lazy(basis::CompositeBasis, indices, op::TimeDependentSum)
    TimeDependentSum(QuantumOpticsBase.coefficients(op),
                     embed_lazy(basis, indices, QuantumOpticsBase.static_operator(op));
                     init_time = current_time(op))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis,
                    indices, op::TimeDependentSum)
    TimeDependentSum(QuantumOpticsBase.coefficients(op),
                     embed_lazy(basis_l, basis_r, indices, QuantumOpticsBase.static_operator(op));
                     init_time = current_time(op))
end

# --- Vector of operators → LazyTensor ---

function embed_lazy(basis::CompositeBasis, indices,
                    operators::AbstractVector{<:AbstractOperator})
    _embed_lazy_check_basis(basis, basis, indices, operators)
    LazyTensor(basis, basis, indices, Tuple(operators))
end

function embed_lazy(basis_l::CompositeBasis, basis_r::CompositeBasis, indices,
                    operators::AbstractVector{<:AbstractOperator})
    _embed_lazy_check_basis(basis_l, basis_r, indices, operators)
    LazyTensor(basis_l, basis_r, indices, Tuple(operators))
end

# ====================== Non-Composite Basis (identity embed) ======================

function embed_lazy(b::Basis, index::Integer, op::AbstractOperator)
    _embed_lazy_check_basis(b, index, op)
    op
end

function embed_lazy(b::Basis, index::Integer, op::LazyTensor)
    _embed_lazy_check_basis(b, index, collect(values(op.operators)))
    op
end

function embed_lazy(b::Basis, index::Integer, op::LazySum)
    _embed_lazy_check_basis(b, index, op.operators)
    op
end

function embed_lazy(b::Basis, index::Integer, op::TimeDependentSum)
    _embed_lazy_check_basis(b, index, QuantumOpticsBase.static_operator(op))
    TimeDependentSum(QuantumOpticsBase.coefficients(op),
                     QuantumOpticsBase.static_operator(op);
                     init_time = current_time(op))
end

function embed_lazy(b::Basis, index::Integer,
                    operators::AbstractVector{<:AbstractOperator})
    _embed_lazy_check_basis(b, index, operators)
    operators
end

# ====================== Convenience (no-index) ======================

embed_lazy(b::Basis, op) = embed_lazy(b, 1, op)
embed_lazy(b::CompositeBasis, op) = embed_lazy(b, 1, op)
