module operators

export AbstractOperator, DataOperator, length, basis, dagger, ishermitian, tensor, embed,
        tr, ptrace, normalize, normalize!, expect, variance,
        exp, permutesystems, identityoperator, dense

import Base: ==, +, -, *, /, ^, length, one, exp, conj, conj!, transpose
import LinearAlgebra: tr, ishermitian
import ..bases: basis, tensor, ptrace, permutesystems,
            samebases, check_samebases, multiplicable
import ..states: dagger, normalize, normalize!

using ..sortedindices, ..bases, ..states


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
`basis_l` and `basis_r` defining the left and right side bases.

For fast time evolution also at least the function
`gemv!(alpha, op::AbstractOperator, x::Ket, beta, result::Ket)` should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract type AbstractOperator{BL<:Basis,BR<:Basis} end

"""
Abstract type for operators with a data field.

This is an abstract type for operators that have a direct matrix representation
stored in their `.data` field.
"""
abstract type DataOperator{BL<:Basis,BR<:Basis} <: AbstractOperator{BL,BR} end


# Common error messages
arithmetic_unary_error(funcname, x::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. dense() or sparse()."))
arithmetic_binary_error(funcname, a::AbstractOperator, b::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. dense() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))

length(a::AbstractOperator) = length(a.basis_l)::Int*length(a.basis_r)::Int
basis(a::AbstractOperator) = (check_samebases(a); a.basis_l)

# Ensure scalar broadcasting
Base.broadcastable(x::AbstractOperator) = Ref(x)

# Arithmetic operations
+(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::AbstractOperator) = addnumbererror()
+(a::AbstractOperator, b::Number) = addnumbererror()

-(a::AbstractOperator) = arithmetic_unary_error("Negation", a)
-(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Subtraction", a, b)
-(a::Number, b::AbstractOperator) = addnumbererror()
-(a::AbstractOperator, b::Number) = addnumbererror()

*(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Multiplication", a, b)
^(a::AbstractOperator, b::Int) = Base.power_by_squaring(a, b)


dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)
Base.adjoint(a::AbstractOperator) = dagger(a)

conj(a::AbstractOperator) = arithmetic_unary_error("Complex conjugate", a)
conj!(a::AbstractOperator) = conj(a::AbstractOperator)

dense(a::AbstractOperator) = arithmetic_unary_error("Conversion to dense", a)

transpose(a::AbstractOperator) = arithmetic_unary_error("Transpose", a)

"""
    ishermitian(op::AbstractOperator)

Check if an operator is Hermitian.
"""
ishermitian(op::AbstractOperator) = arithmetic_unary_error(ishermitian, op)


"""
    tensor(x::AbstractOperator, y::AbstractOperator, z::AbstractOperator...)

Tensor product ``\\hat{x}⊗\\hat{y}⊗\\hat{z}⊗…`` of the given operators.
"""
tensor(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Tensor product", a, b)
tensor(op::AbstractOperator) = op
tensor(operators::AbstractOperator...) = reduce(tensor, operators)


"""
    embed(basis1[, basis2], indices::Vector, operators::Vector)

Tensor product of operators where missing indices are filled up with identity operators.
"""
function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
               indices::Vector{Int}, operators::Vector{T}) where T<:AbstractOperator
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N
    @assert length(indices) >= length(operators)
    op_b_len = Int[isa(op.basis_l, CompositeBasis) ? length(op.basis_l.bases) : 1 for op=operators]

    # Neglect indices covered by composite operators
    k = 1
    new_idx = Int[]
    for i=1:length(operators)
        push!(new_idx, indices[k])
        ptr_index = filter(p->!(p ∈ indices[k:k+op_b_len[i]-1]), 1:N)
        operators[i].basis_l == ptrace(basis_l, ptr_index) || throw(bases.IncompatibleBases())
        operators[i].basis_r == ptrace(basis_r, ptr_index) || throw(bases.IncompatibleBases())
        k += op_b_len[i]
    end
    # Recursively lower index to fit length
    for i=1:length(operators)-1
        new_idx[i+1:end] .-= (op_b_len[i] - 1)
    end

    n = N-sum(op_b_len) + length(operators)
    # Assign operators according to their position
    out = Array{AbstractOperator,1}(undef, n)
    for i=1:length(operators)
        out[new_idx[i]] = operators[i]
    end
    # Find positions and assign identities
    id_idx = filter(p->!(p ∈ indices), 1:N)
    new_id_idx = filter(p->!(p ∈ new_idx), 1:n)
    @assert length(id_idx) == length(new_id_idx)
    for i=1:length(id_idx)
        out[new_id_idx[i]] = identityoperator(T, basis_l.bases[id_idx[i]], basis_r.bases[id_idx[i]])
    end

    return tensor(out...)
end
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Int, op::AbstractOperator) = embed(basis_l, basis_r, Int[index], [op])
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Vector{Int}, op::AbstractOperator) = embed(basis_l, basis_r, index, [op])
embed(basis::CompositeBasis, index::Int, op::AbstractOperator) = embed(basis, basis, Int[index], [op])
embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{T}) where {T<:AbstractOperator} = embed(basis, basis, indices, operators)
embed(basis::CompositeBasis, index::Vector{Int}, op::AbstractOperator) = embed(basis, basis, index, [op])

"""
    embed(basis1[, basis2], operators::Dict)

`operators` is a dictionary `Dict{Vector{Int}, AbstractOperator}`. The integer vector
specifies in which subsystems the corresponding operator is defined.
"""
function embed(basis_l::CompositeBasis, basis_r::CompositeBasis,
               operators::Dict{Vector{Int}, T}) where T<:AbstractOperator
    @assert length(basis_l.bases) == length(basis_r.bases)
    N = length(basis_l.bases)
    if length(operators) == 0
        return identityoperator(T, basis_l, basis_r)
    end
    indices, operator_list = zip(operators...)
    operator_list = [operator_list...;]
    indices_flat = [indices...;]
    start_indices_flat = [i[1] for i in indices]
    complement_indices_flat = Int[i for i=1:N if i ∉ indices_flat]
    operators_flat = T[]
    if all([minimum(I):maximum(I);]==I for I in indices)
        for i in 1:N
            if i in complement_indices_flat
                push!(operators_flat, identityoperator(T, basis_l.bases[i], basis_r.bases[i]))
            elseif i in start_indices_flat
                push!(operators_flat, operator_list[indexin(i, start_indices_flat)[1]])
            end
        end
        return tensor(operators_flat...)
    else
        complement_operators = [identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i in complement_indices_flat]
        op = tensor([operator_list; complement_operators]...)
        perm = sortperm([indices_flat; complement_indices_flat])
        return permutesystems(op, perm)
    end
end
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int, T}; kwargs...) where {T<:AbstractOperator} = embed(basis_l, basis_r, Dict([i]=>op_i for (i, op_i) in operators); kwargs...)
embed(basis::CompositeBasis, operators::Dict{Int, T}; kwargs...) where {T<:AbstractOperator} = embed(basis, basis, operators; kwargs...)
embed(basis::CompositeBasis, operators::Dict{Vector{Int}, T}; kwargs...) where {T<:AbstractOperator} = embed(basis, basis, operators; kwargs...)


"""
    tr(x::AbstractOperator)

Trace of the given operator.
"""
tr(x::AbstractOperator) = arithmetic_unary_error("Trace", x)

ptrace(a::AbstractOperator, index::Vector{Int}) = arithmetic_unary_error("Partial trace", a)

"""
    normalize(op)

Return the normalized operator so that its `tr(op)` is one.
"""
normalize(op::AbstractOperator) = op/tr(op)

"""
    normalize!(op)

In-place normalization of the given operator so that its `tr(x)` is one.
"""
normalize!(op::AbstractOperator) = throw(ArgumentError("normalize! is not defined for this type of operator: $(typeof(op)).\n You may have to fall back to the non-inplace version 'normalize()'."))

"""
    expect(op, state)

Expectation value of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
expect(op::AbstractOperator{B,B}, state::Ket{B}) where B<:Basis = state.data' * (op * state).data
expect(op::AbstractOperator{B1,B2}, state::AbstractOperator{B2,B2}) where {B1<:Basis,B2<:Basis} = tr(op*state)

"""
    expect(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number.
"""
function expect(indices::Vector{Int}, op::AbstractOperator{B1,B2}, state::AbstractOperator{B3,B3}) where {B1<:Basis,B2<:Basis,B3<:CompositeBasis}
    N = length(state.basis_l.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
function expect(indices::Vector{Int}, op::AbstractOperator{B,B}, state::Ket{B2}) where {B<:Basis,B2<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
expect(index::Int, op::AbstractOperator, state) = expect([index], op, state)
expect(op::AbstractOperator, states::Vector) = [expect(op, state) for state=states]
expect(indices::Vector{Int}, op::AbstractOperator, states::Vector) = [expect(indices, op, state) for state=states]

"""
    variance(op, state)

Variance of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function variance(op::AbstractOperator{B,B}, state::Ket{B}) where B<:Basis
    x = op*state
    state.data'*(op*x).data - (state.data'*x.data)^2
end
function variance(op::AbstractOperator{B,B}, state::AbstractOperator{B,B}) where B<:Basis
    expect(op*op, state) - expect(op, state)^2
end

"""
    variance(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number
"""
function variance(indices::Vector{Int}, op::AbstractOperator{B,B}, state::AbstractOperator{BC,BC}) where {B<:Basis,BC<:CompositeBasis}
    N = length(state.basis_l.shape)
    indices_ = sortedindices.complement(N, indices)
    variance(op, ptrace(state, indices_))
end
function variance(indices::Vector{Int}, op::AbstractOperator{B,B}, state::Ket{BC}) where {B<:Basis,BC<:CompositeBasis}
    N = length(state.basis.shape)
    indices_ = sortedindices.complement(N, indices)
    variance(op, ptrace(state, indices_))
end
variance(index::Int, op::AbstractOperator, state) = variance([index], op, state)
variance(op::AbstractOperator, states::Vector) = [variance(op, state) for state=states]
variance(indices::Vector{Int}, op::AbstractOperator, states::Vector) = [variance(indices, op, state) for state=states]


"""
    exp(op::AbstractOperator)

Operator exponential.
"""
exp(op::AbstractOperator) = throw(ArgumentError("exp() is not defined for this type of operator: $(typeof(op)).\nTry to convert to dense operator first with dense()."))

permutesystems(a::AbstractOperator, perm::Vector{Int}) = arithmetic_unary_error("Permutations of subsystems", a)

"""
    identityoperator(a::Basis[, b::Basis])

Return an identityoperator in the given bases.
"""
identityoperator(::Type{T}, b1::Basis, b2::Basis) where {T<:AbstractOperator} = throw(ArgumentError("Identity operator not defined for operator type $T."))
identityoperator(::Type{T}, b::Basis) where {T<:AbstractOperator} = identityoperator(T, b, b)
identityoperator(op::T) where {T<:AbstractOperator} = identityoperator(T, op.basis_l, op.basis_r)

one(b::Basis) = identityoperator(b)
one(op::AbstractOperator) = identityoperator(op)


# Fast in-place multiplication
"""
    gemv!(alpha, a, b, beta, result)

Fast in-place multiplication of operators with state vectors. It
implements the relation `result = beta*result + alpha*a*b`.
Here, `alpha` and `beta` are complex numbers, while `result` and either `a`
or `b` are state vectors while the other one can be of any operator type.
"""
gemv!() = error("Not Implemented.")

"""
    gemm!(alpha, a, b, beta, result)

Fast in-place multiplication of of operators with DenseOperators. It
implements the relation `result = beta*result + alpha*a*b`.
Here, `alpha` and `beta` are complex numbers, while `result` and either `a`
or `b` are dense operators while the other one can be of any operator type.
"""
gemm!() = error("Not Implemented.")


# Helper functions to check validity of arguments
function check_ptrace_arguments(a::AbstractOperator, indices::Vector{Int})
    if !isa(a.basis_l, CompositeBasis) || !isa(a.basis_r, CompositeBasis)
        throw(ArgumentError("Partial trace can only be applied onto operators with composite bases."))
    end
    rank = length(a.basis_l.shape)
    if rank != length(a.basis_r.shape)
        throw(ArgumentError("Partial trace can only be applied onto operators wich have the same number of subsystems in the left basis and right basis."))
    end
    if rank == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    sortedindices.check_indices(length(a.basis_l.shape), indices)
    for i=indices
        if a.basis_l.shape[i] != a.basis_r.shape[i]
            throw(ArgumentError("Partial trace can only be applied onto subsystems that have the same left and right dimension."))
        end
    end
end
function check_ptrace_arguments(a::StateVector, indices::Vector{Int})
    if length(basis(a).shape) == length(indices)
        throw(ArgumentError("Partial trace can't be used to trace out all subsystems - use tr() instead."))
    end
    sortedindices.check_indices(length(basis(a).shape), indices)
end

samebases(a::AbstractOperator) = samebases(a.basis_l, a.basis_r)::Bool
samebases(a::AbstractOperator, b::AbstractOperator) = samebases(a.basis_l, b.basis_l)::Bool && samebases(a.basis_r, b.basis_r)::Bool
check_samebases(a::AbstractOperator) = check_samebases(a.basis_l, a.basis_r)

multiplicable(a::AbstractOperator, b::Ket) = multiplicable(a.basis_r, b.basis)
multiplicable(a::Bra, b::AbstractOperator) = multiplicable(a.basis, b.basis_l)
multiplicable(a::AbstractOperator, b::AbstractOperator) = multiplicable(a.basis_r, b.basis_l)

end # module
