module operators_dense

export Operator, dense, projector, dm, mul!

import Base: ==, +, -, *, /
import LinearAlgebra: mul!
import ..operators

using LinearAlgebra, Base.Cartesian, SparseArrays
using ..bases, ..states, ..operators

const OperatorDataType = Union{Matrix{ComplexF64},SparseMatrixCSC{ComplexF64,Int}}

"""
    Operator(b1[, b2, data])

Array implementation of Operator.

The matrix consisting of complex floats is stored in the `data` field.
"""
mutable struct Operator{BL<:Basis,BR<:Basis,T<:OperatorDataType} <: AbstractOperator{BL,BR}
    basis_l::BL
    basis_r::BR
    data::T
    function Operator{BL,BR,T}(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:OperatorDataType}
        @assert length(b1) == size(data, 1) && length(b2) == size(data, 2)
        new(b1, b2, data)
    end
end

Operator(b1::BL, b2::BR, data::T) where {BL<:Basis,BR<:Basis,T<:OperatorDataType} = Operator{BL,BR,T}(b1, b2, data)
Operator(b::Basis, data::OperatorDataType) = Operator(b, b, data)
Operator(b1::Basis, b2::Basis) = Operator(b1, b2, zeros(ComplexF64, length(b1), length(b2)))
Operator(b::Basis) = Operator(b, b)
Operator(op::AbstractOperator) = dense(op)

Base.copy(x::Operator{BL,BR,T}) where {BL<:Basis,BR<:Basis,T<:OperatorDataType} = Operator{BL,BR,T}(x.basis_l, x.basis_r, copy(x.data))

"""
    dense(op::AbstractOperator)

Convert an arbitrary Operator into a [`Operator`](@ref) with a dense `data` field.
"""
operators.dense(x::Operator{BL,BR,T}) where {BL<:Basis,BR<:Basis,T<:OperatorDataType} = Operator{BL,BR}(x.basis_l, x.basis_r, Matrix(x.data))

==(x::Operator, y::Operator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


# Arithmetic operations
+(a::T, b::T) where T<:Operator = Operator(a.basis_l, a.basis_r, a.data+b.data)

-(a::Operator) = Operator(a.basis_l, a.basis_r, -a.data)
-(a::T, b::T) where T<:Operator = Operator(a.basis_l, a.basis_r, a.data-b.data)

*(a::Operator{BL,BR}, b::Ket{BR}) where {BL<:Basis,BR<:Basis} = Ket(a.basis_l, a.data*b.data)
*(a::Bra{BL}, b::Operator{BL,BR}) where {BL<:Basis,BR<:Basis} = Bra(b.basis_r, transpose(b.data)*a.data)
*(a::Operator{BL1,BR}, b::Operator{BR,BR2}) where {BL1<:Basis,BR<:Basis,BR2<:Basis} = Operator(a.basis_l, b.basis_r, a.data*b.data)
*(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::Operator) = Operator(b.basis_l, b.basis_r, complex(a)*b.data)
function *(op1::AbstractOperator{BL1,BR}, op2::Operator{BR,BR2}) where {BL1<:Basis,BR<:Basis,BR2<:Basis}
    result = Operator{BL1,BR2}(op1.basis_l, op2.basis_r)
    mul!(result, op1, op2)
    return result
end
function *(op1::Operator{BL1,BR}, op2::AbstractOperator{BR,BR2}) where {BL1<:Basis,BR<:Basis,BR2<:Basis}
    result = Operator{BL1,BR2}(op1.basis_l, op2.basis_r)
    mul!(result, op1, op2)
    return result
end
function *(op::AbstractOperator{BL,BR}, psi::Ket{BR}) where {BL<:Basis,BR<:Basis}
    result = Ket{BL}(op.basis_l)
    mul!(result, op, psi)
    return result
end
function *(psi::Bra{BL}, op::AbstractOperator{BL,BR}) where {BR<:Basis,BL<:Basis}
    result = Bra{BR}(op.basis_r)
    mul!(result, psi, op)
    return result
end

/(a::Operator, b::Number) = Operator(a.basis_l, a.basis_r, a.data/complex(b))


operators.dagger(x::Operator) = Operator(x.basis_r, x.basis_l, x.data')

operators.ishermitian(A::Operator) = (A.basis_l == A.basis_r) && ishermitian(A.data)

operators.tensor(a::Operator, b::Operator) = Operator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(b.data, a.data))

operators.conj(a::Operator) = Operator(a.basis_l, a.basis_r, conj.(a.data))
operators.conj!(a::Operator) = conj!.(a.data)

"""
    tensor(x::Ket, y::Bra)

Outer product ``|x⟩⟨y|`` of the given states.
"""
operators.tensor(a::Ket{BL}, b::Bra{BR}) where {BL<:Basis,BR<:Basis} = Operator{BL,BR,T<:Matrix{ComplexF64}}(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))


operators.tr(op::Operator{B,B}) where B<:Basis = tr(op.data)

function operators.ptrace(a::Operator, indices::Vector{Int})
    operators.check_ptrace_arguments(a, indices)
    rank = length(a.basis_l.shape)
    result = _ptrace(Val{rank}, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return Operator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

function operators.ptrace(psi::Ket, indices::Vector{Int})
    operators.check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_ket(Val{rank}, psi.data, b.shape, indices)
    return Operator(b_, b_, result)
end
function operators.ptrace(psi::Bra, indices::Vector{Int})
    operators.check_ptrace_arguments(psi, indices)
    b = basis(psi)
    b_ = ptrace(b, indices)
    rank = length(b.shape)
    result = _ptrace_bra(Val{rank}, psi.data, b.shape, indices)
    return Operator(b_, b_, result)
end

operators.normalize!(op::Operator) = (rmul!(op.data, 1.0/tr(op)); nothing)

function operators.expect(op::Operator{B,B}, state::Ket{B}) where B<:Basis
    state.data' * op.data * state.data
end

function operators.expect(op::Operator{BL,BR,T}, state::AbstractOperator{BR,BL}) where {BL<:Basis,BR<:Basis,T<:Matrix{ComplexF64}}
    result = ComplexF64(0.)
    @inbounds for i=1:size(op.data, 1), j=1:size(op.data,2)
        result += op.data[i,j]*state.data[j,i]
    end
    result
end

function operators.exp(op::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}}
    return Operator{B,B,T}(op.basis_l, op.basis_r, exp(op.data))
end

function operators.permutesystems(a::Operator{BL,BR,T}, perm::Vector{Int}) where {BL<:CompositeBasis,BR<:CompositeBasis,T<:Matrix{ComplexF64}}
    @assert length(a.basis_l.bases) == length(a.basis_r.bases) == length(perm)
    @assert isperm(perm)
    data = reshape(a.data, [a.basis_l.shape; a.basis_r.shape]...)
    data = permutedims(data, [perm; perm .+ length(perm)])
    data = reshape(data, length(a.basis_l), length(a.basis_r))
    Operator(permutesystems(a.basis_l, perm), permutesystems(a.basis_r, perm), data)
end

operators.identityoperator(::Type{Operator{BL,BR,T}}, b1::BL, b2::BR) where {BL<:Basis,BR<:Basis,T<:OperatorDataType} = Operator{BL,BR,T}(b1, b2, T(I, length(b1), length(b2)))
operators.identityoperator(::Type{Operator{BL1,BR1,T}}, b1::BL2,
        b2::BR2) where {BL1<:Basis,BR1<:Basis,BL2<:Basis,BR2<:Basis,T<:OperatorDataType} =
    Operator{BL2,BR2,T}(b1, b2, T(I, length(b1), length(b2)))

"""
    projector(a::Ket, b::Bra)

Projection operator ``|a⟩⟨b|``.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
"""
    projector(a::Ket)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Ket) = tensor(a, dagger(a))
"""
    projector(a::Bra)

Projection operator ``|a⟩⟨a|``.
"""
projector(a::Bra) = tensor(dagger(a), a)

"""
    dm(a::StateVector)

Create density matrix ``|a⟩⟨a|``. Same as `projector(a)`.
"""
dm(x::Ket) = tensor(x, dagger(x))
dm(x::Bra) = tensor(dagger(x), x)


# Partial trace implementation for dense operators.
function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[1] = 1
    for m=2:N
        S[m] = S[m-1]*shape[m-1]
    end
    return S
end

# Dense operator version
@generated function _ptrace(::Type{Val{RANK}}, a::Matrix{ComplexF64},
                            shape_l::Vector{Int}, shape_r::Vector{Int},
                            indices::Vector{Int}) where RANK
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = copy(shape_l)
        result_shape_l[indices] .= 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = copy(shape_r)
        result_shape_r[indices] .= 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(ComplexF64, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

@generated function _ptrace_ket(::Type{Val{RANK}}, a::Vector{ComplexF64},
                            shape::Vector{Int}, indices::Vector{Int}) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        result_shape[indices] .= 1
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(ComplexF64, N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += a[Il_0]*conj(a[Ir_0])
            end
        end
        return result
    end
end

@generated function _ptrace_bra(::Type{Val{RANK}}, a::Vector{ComplexF64},
                            shape::Vector{Int}, indices::Vector{Int}) where RANK
    return quote
        a_strides = _strides(shape)
        result_shape = copy(shape)
        result_shape[indices] .= 1
        result_strides = _strides(result_shape)
        N_result = prod(result_shape)
        result = zeros(ComplexF64, N_result, N_result)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides[d]; if !(d in indices) Jr_d+=result_strides[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides[k]; continue end)) (k->(Il_k+=a_strides[k]; if !(k in indices) Jl_k+=result_strides[k] end)) begin
                result[Jl_0, Jr_0] += conj(a[Il_0])*a[Ir_0]
            end
        end
        return result
    end
end

# Fast in-place multiplication
mul!(result::Operator{BL,BR,T1}, a::Operator{BL,BR2,T2}, b::Operator{BR2,BR,T1}) where {BL<:Basis,BR<:Basis,BR2<:Basis,T1<:OperatorDataType,T2<:OperatorDataType} =
    mul!(result.data, a.data, b.data)
mul!(result::Operator{BL,BR,T1}, a::Operator{BL,BR2,T2}, b::Operator{BR2,BR,T1}, alpha::Number, beta::Number) where {BL<:Basis,BR<:Basis,BR2<:Basis,T1<:OperatorDataType,T2<:OperatorDataType} =
    mul!(result.data, a.data, b.data)

mul!(result::Ket{B}, a::Operator{B,BR}, b::Ket{BR}) where {B<:Basis,BR<:Basis} =
    mul!(result, a.data, b.data)
mul!(result::Ket{B}, a::Operator{B,BR}, b::Ket{BR}, alpha::Number, beta::Number) where {B<:Basis,BR<:Basis} =
    mul!(result, a.data, b.data, alpha, beta)

mul!(result::Bra{B}, a::Bra{BL}, b::Operator{BL,B}) where {B<:Basis,BL<:Basis} =
    mul!(result, a.data, b.data)
mul!(result::Bra{B}, a::Bra{BL}, b::Operator{BL,B}, alpha::Number, beta::Number) where {B<:Basis,BL<:Basis} =
    mul!(result, a.data, b.data, alpha, beta)
# operators.gemm!(alpha, a::Matrix{ComplexF64}, b::Matrix{ComplexF64}, beta, result::Matrix{ComplexF64}) = BLAS.gemm!('N', 'N', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
# operators.gemv!(alpha, a::Matrix{ComplexF64}, b::Vector{ComplexF64}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('N', convert(ComplexF64, alpha), a, b, convert(ComplexF64, beta), result)
# operators.gemv!(alpha, a::Vector{ComplexF64}, b::Matrix{ComplexF64}, beta, result::Vector{ComplexF64}) = BLAS.gemv!('T', convert(ComplexF64, alpha), b, a, convert(ComplexF64, beta), result)

# operators.gemm!(alpha, a::DenseOperator, b::DenseOperator, beta, result::DenseOperator) = operators.gemm!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
# operators.gemv!(alpha, a::DenseOperator, b::Ket, beta, result::Ket) = operators.gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)
# operators.gemv!(alpha, a::Bra, b::DenseOperator, beta, result::Bra) = operators.gemv!(convert(ComplexF64, alpha), a.data, b.data, convert(ComplexF64, beta), result.data)


# # Multiplication for Operators in terms of their gemv! implementation
# function operators.gemm!(alpha, M::AbstractOperator, b::DenseOperator, beta, result::DenseOperator)
#     for i=1:size(b.data, 2)
#         bket = Ket(b.basis_l, b.data[:,i])
#         resultket = Ket(M.basis_l, result.data[:,i])
#         operators.gemv!(alpha, M, bket, beta, resultket)
#         result.data[:,i] = resultket.data
#     end
# end
#
# function operators.gemm!(alpha, b::DenseOperator, M::AbstractOperator, beta, result::DenseOperator)
#     for i=1:size(b.data, 1)
#         bbra = Bra(b.basis_r, vec(b.data[i,:]))
#         resultbra = Bra(M.basis_r, vec(result.data[i,:]))
#         operators.gemv!(alpha, bbra, M, beta, resultbra)
#         result.data[i,:] = resultbra.data
#     end
# end

end # module
