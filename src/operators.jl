module operators

import Base: trace, ==, +, -, *, /, ishermitian
import ..bases: tensor, ptrace, permutesystems
import ..states: dagger, normalize, normalize!

using ..sortedindices, ..bases, ..states

export Operator,
       dagger, identityoperator,
       trace, ptrace, normalize, normalize!, expect, variance,
       tensor, permutesystems, embed


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
`basis_l` and `basis_r` defining the left and right side bases.

For fast time evolution also at least the function
`gemv!(alpha, op::Operator, x::Ket, beta, result::Ket)` should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract Operator

# Common error messages
arithmetic_unary_error(funcname, x::Operator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. full() or sparse()."))
arithmetic_binary_error(funcname, a::Operator, b::Operator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. full() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))


# Arithmetic operations
*(a::Operator, b::Operator) = arithmetic_binary_error("Multiplication", a, b)

+(a::Operator, b::Operator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::Operator) = addnumbererror()
+(a::Operator, b::Number) = addnumbererror()

-(a::Operator) = arithmetic_unary_error("Negation", a)
-(a::Operator, b::Operator) = arithmetic_binary_error("Subtraction", a, b)
-(a::Number, b::Operator) = addnumbererror()
-(a::Operator, b::Number) = addnumbererror()

bases.basis(a::Operator) = (check_samebases(a); a.basis_l)

dagger(a::Operator) = arithmetic_unary_error("Hermitian conjugate", a)

"""
    identityoperator(a::Basis[, b::Basis])

Return an identityoperator in the given bases.
"""
identityoperator{T<:Operator}(::Type{T}, b1::Basis, b2::Basis) = throw(ArgumentError("Identity operator not defined for operator type $T."))
identityoperator{T<:Operator}(::Type{T}, b::Basis) = identityoperator(T, b, b)
identityoperator{T<:Operator}(op::T) = identityoperator(T, op.basis_l, op.basis_r)

Base.one(b::Basis) = identityoperator(b)
Base.one(op::Operator) = identityoperator(op)

"""
    trace(x::Operator)

Trace of the given operator.
"""
trace(x::Operator) = arithmetic_unary_error("Trace", x)

ptrace(a::Operator, index::Vector{Int}) = arithmetic_unary_error("Partial trace", a)

"""
    normalize(op)

Return the normalized operator so that its `trace(op)` is one.
"""
normalize(op::Operator) = op/trace(op)

"""
    normalize!(op)

In-place normalization of the given operator so that its `trace(x)` is one.
"""
normalize!(op::Operator) = throw(ArgumentError("normalize! is not defined for this type of operator: $(typeof(op)).\n You may have to fall back to the non-inplace version 'normalize()'."))

"""
    expect(op, state)

Expectation value of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
expect(op::Operator, state::Ket) = dagger(state)*(op*state)
expect(op::Operator, state::Operator) = trace(op*state)

"""
    expect(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number.
"""
function expect(indices::Vector{Int}, op::Operator, state::Operator)
    N = length(state.basis_l.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
function expect(indices::Vector{Int}, op::Operator, state::Ket)
    N = length(state.basis.shape)
    indices_ = sortedindices.complement(N, indices)
    expect(op, ptrace(state, indices_))
end
expect(index::Int, op::Operator, state) = expect([index], op, state)
expect(op::Operator, states::Vector) = [expect(op, state) for state=states]
expect(indices::Vector{Int}, op::Operator, states::Vector) = [expect(indices, op, state) for state=states]

"""
    variance(op, state)

Variance of the given operator `op` for the specified `state`.

`state` can either be a (density) operator or a ket.
"""
function variance(op::Operator, state::Ket)
    x = op*state
    stateT = dagger(state)
    stateT*op*x - (stateT*x)^2
end
function variance(op::Operator, state::Operator)
    expect(op*op, state) - expect(op, state)^2
end

"""
    variance(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number
"""
function variance(indices::Vector{Int}, op::Operator, state::Operator)
    N = length(state.basis_l.shape)
    indices_ = sortedindices.complement(N, indices)
    variance(op, ptrace(state, indices_))
end
function variance(indices::Vector{Int}, op::Operator, state::Ket)
    N = length(state.basis.shape)
    indices_ = sortedindices.complement(N, indices)
    variance(op, ptrace(state, indices_))
end
variance(index::Int, op::Operator, state) = variance([index], op, state)
variance(op::Operator, states::Vector) = [variance(op, state) for state=states]
variance(indices::Vector{Int}, op::Operator, states::Vector) = [variance(indices, op, state) for state=states]


"""
    tensor(x::Operator, y::Operator, z::Operator...)

Tensor product ``\\hat{x}⊗\\hat{y}⊗\\hat{z}⊗…`` of the given operators.
"""
tensor(a::Operator, b::Operator) = arithmetic_binary_error("Tensor product", a, b)
bases.tensor(op::Operator) = op
bases.tensor(operators::Operator...) = reduce(tensor, operators)

permutesystems(a::Operator, perm::Vector{Int}) = arithmetic_unary_error("Permutations of subsystems", a)

"""
Vector of indices that are not in the given vector.
"""
complement(N::Int, indices) = Int[i for i=1:N if i ∉ indices]

"""
    embed(basis1[, basis2], indices::Vector, operators::Vector)

Tensor product of operators where missing indices are filled up with identity operators.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            indices::Vector{Int}, operators::Vector{T})
    N = length(basis_l.bases)
    @assert length(basis_r.bases) == N
    @assert length(indices) == length(operators)
    sortedindices.check_indices(N, indices)
    tensor([i ∈ indices ? operators[findfirst(indices, i)] : identityoperator(T, basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
end
embed(basis_l::CompositeBasis, basis_r::CompositeBasis, index::Int, op::Operator) = embed(basis_l, basis_r, Int[index], [op])
embed(basis::CompositeBasis, index::Int, op::Operator) = embed(basis, basis, Int[index], [op])
embed{T<:Operator}(basis::CompositeBasis, indices::Vector{Int}, operators::Vector{T}) = embed(basis, basis, indices, operators)

"""
    embed(basis1[, basis2], operators::Dict)

`operators` is a dictionary `Dict{Vector{Int}, Operator}`. The integer vector
specifies in which subsystems the corresponding operator is defined.
"""
function embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                            operators::Dict{Vector{Int}, T})
    @assert length(basis_l.bases) == length(basis_r.bases)
    N = length(basis_l.bases)
    if length(operators) == 0
        return identityoperator(T, basis_l, basis_r)
    end
    indices, operator_list = zip(operators...)
    operator_list = [operator_list...;]
    indices_flat = [indices...;]
    start_indices_flat = [i[1] for i in indices]
    complement_indices_flat = complement(N, indices_flat)
    operators_flat = T[]
    if all([minimum(I):maximum(I);]==I for I in indices)
        for i in 1:N
            if i in complement_indices_flat
                push!(operators_flat, identityoperator(T, basis_l.bases[i], basis_r.bases[i]))
            elseif i in start_indices_flat
                push!(operators_flat, operator_list[findfirst(start_indices_flat, i)])
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
embed{T<:Operator}(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis_l, basis_r, Dict([i]=>op_i for (i, op_i) in operators); kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Int, T}; kwargs...) = embed(basis, basis, operators; kwargs...)
embed{T<:Operator}(basis::CompositeBasis, operators::Dict{Vector{Int}, T}; kwargs...) = embed(basis, basis, operators; kwargs...)

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
function check_ptrace_arguments(a::Operator, indices::Vector{Int})
    @assert length(a.basis_l.shape) == length(a.basis_r.shape)
    sortedindices.check_indices(length(a.basis_l.shape), indices)
    for i=indices
        if a.basis_l.shape[i] != a.basis_r.shape[i]
            throw(ArgumentError("Partial trace can only be applied onto subsystems that have the same left and right dimension."))
        end
    end
end

bases.samebases(a::Operator) = samebases(a.basis_l, a.basis_r)
bases.samebases(a::Operator, b::Operator) = samebases(a.basis_l, b.basis_l) && samebases(a.basis_r, b.basis_r)
bases.check_samebases(a::Operator) = check_samebases(a.basis_l, a.basis_r)

bases.multiplicable(a::Operator, b::Ket) = multiplicable(a.basis_r, b.basis)
bases.multiplicable(a::Bra, b::Operator) = multiplicable(a.basis, b.basis_l)
bases.multiplicable(a::Operator, b::Operator) = multiplicable(a.basis_r, b.basis_l)

"""
    ishermitian(op::Operator)

Check if an operator is Hermitian.
"""
ishermitian(op::Operator) = arithmetic_unary_error(ishermitian, op)

"""
    expm(op::Operator)

Operator exponential.
"""
Base.expm(op::Operator) = throw(ArgumentError("expm() is not defined for this type of operator: $(typeof(op)).\nTry to convert to dense operator first with full()."))

end # module
