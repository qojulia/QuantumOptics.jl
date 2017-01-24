module operators

import Base: ==, +, -, *, /
import ..states

using Base.LinAlg.BLAS
using Base.Cartesian

using ..bases, ..states

export Operator, DenseOperator,
       tensor, dagger, projector,
       expect, identityoperator, dense_identityoperator,
       embed


"""
Abstract base class for all operators.

All deriving operator classes have to define the fields
basis_l and basis_r defining the left and right side bases.

For fast time evolution also at least the function
gemv!(alpha, op::Operator, x::Ket, beta, result::Ket) should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract Operator

"""
Dense array implementation of Operator.

The matrix consisting of complex floats is stored in the data field.
"""
type DenseOperator <: Operator
    basis_l::Basis
    basis_r::Basis
    data::Matrix{Complex{Float64}}
    DenseOperator(b1::Basis, b2::Basis, data) = length(b1) == size(data, 1) && length(b2) == size(data, 2) ? new(b1, b2, data) : throw(DimensionMismatch())
end

DenseOperator(b::Basis, data) = DenseOperator(b, b, data)
DenseOperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, zeros(Complex, length(b1), length(b2)))
DenseOperator(b::Basis) = DenseOperator(b, b)
DenseOperator(op::Operator) = full(op)

"""
Converting an arbitrary Operator into a DenseOperator.
"""
Base.copy(x::DenseOperator) = deepcopy(x)
Base.full(x::DenseOperator) = deepcopy(x)
Base.full(op::Operator) = op*dense_identityoperator(op.basis_r)

Base.eltype(x::Operator) = Complex128
Base.zero{T<:Operator}(x::T) = T(x.basis_l, x.basis_r)

==(x::DenseOperator, y::DenseOperator) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && (x.data == y.data)


# Arithmetic operations for dense DenseOperators
check_samebases(a::Operator, b::Operator) = ((a.basis_l!=b.basis_l) || (a.basis_r!=b.basis_r) ? throw(IncompatibleBases()) : nothing)

*(a::DenseOperator, b::Ket) = (check_multiplicable(a.basis_r, b.basis); Ket(a.basis_l, a.data*b.data))
*(a::Bra, b::DenseOperator) = (check_multiplicable(a.basis, b.basis_l); Bra(b.basis_r, b.data.'*a.data))
*(a::DenseOperator, b::DenseOperator) = (check_multiplicable(a.basis_r, b.basis_l); DenseOperator(a.basis_l, b.basis_r, a.data*b.data))
*(a::DenseOperator, b::Number) = DenseOperator(a.basis_l, a.basis_r, complex(b)*a.data)
*(a::Number, b::DenseOperator) = DenseOperator(b.basis_l, b.basis_r, complex(a)*b.data)

/(a::DenseOperator, b::Number) = DenseOperator(a.basis_l, a.basis_r, a.data/complex(b))

+(a::DenseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::DenseOperator) = DenseOperator(a.basis_l, a.basis_r, -a.data)
-(a::DenseOperator, b::DenseOperator) = (check_samebases(a,b); DenseOperator(a.basis_l, a.basis_r, a.data-b.data))


# Fast in-place multiplication implementations
gemm!{T<:Complex}(alpha::T, a::Matrix{T}, b::Matrix{T}, beta::T, result::Matrix{T}) = BLAS.gemm!('N', 'N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Matrix{T}, b::Vector{T}, beta::T, result::Vector{T}) = BLAS.gemv!('N', alpha, a, b, beta, result)
gemv!{T<:Complex}(alpha::T, a::Vector{T}, b::Matrix{T}, beta::T, result::Vector{T}) = BLAS.gemv!('T', alpha, b, a, beta, result)

gemm!(alpha, a::DenseOperator, b::DenseOperator, beta, result::DenseOperator) = gemm!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::DenseOperator, b::Ket, beta, result::Ket) = gemv!(alpha, a.data, b.data, beta, result.data)
gemv!(alpha, a::Bra, b::DenseOperator, beta, result::Bra) = gemv!(alpha, a.data, b.data, beta, result.data)


"""
Tensor product of operators.
"""
bases.tensor(a::DenseOperator, b::DenseOperator) = DenseOperator(tensor(a.basis_l, b.basis_l), tensor(a.basis_r, b.basis_r), kron(a.data, b.data))
bases.tensor(ops::DenseOperator...) = reduce(tensor, ops)

"""
Tensor product of a ket and a bra results in an operator.
"""
bases.tensor(a::Ket, b::Bra) = DenseOperator(a.basis, b.basis, reshape(kron(b.data, a.data), prod(a.basis.shape), prod(b.basis.shape)))

"""
Hermitian conjugate of the given operator.
"""
states.dagger(x::DenseOperator) = DenseOperator(x.basis_r, x.basis_l, x.data')


"""
Create a projection operator.
"""
projector(a::Ket, b::Bra) = tensor(a, b)
projector(a::Ket) = tensor(a, dagger(a))
projector(a::Bra) = tensor(dagger(a), a)

"""
p-norm of given operator.
"""
Base.norm(op::DenseOperator, p) = norm(op.data, p)

"""
Trace of given operator.
"""
Base.trace(op::DenseOperator) = trace(op.data)

"""
Normalized copy of given operator (trace is 1.).
"""
states.normalize(op::DenseOperator) = op/trace(op)

"""
Normalize the given operator.
"""
function states.normalize!(op::DenseOperator)
    u = 1./trace(op)
    for j=1:size(op.data,2), i=1:size(op.data,1)
        op.data[i,j] *= u
    end
    return op
end


"""
Operator exponential.
"""
function Base.expm(op::DenseOperator)
    if !multiplicable(op.basis_r, op.basis_l)
        throw(ArgumentError("Operator has to be multiplicable with itself."))
    end
    return DenseOperator(op.basis_l, op.basis_r, expm(op.data))
end


"""
Expectation value of the given operator for the specified state(s).
"""
expect(op::Operator, state::DenseOperator) = trace(op*state)
expect(op::Operator, states::Vector{DenseOperator}) = [expect(op, state) for state=states]
expect(op::Operator, state::Ket) = dagger(state)*(op*state)
expect(op::Operator, states::Vector{Ket}) = [expect(op, state) for state=states]

"""
Identity operator.
"""
dense_identityoperator(b::Basis) = DenseOperator(b, b, eye(Complex, length(b)))
dense_identityoperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, eye(Complex, length(b1), length(b2)))
dense_identityoperator(op::Operator) = dense_identityoperator(op.basis_l, op.basis_r)
identityoperator(op::DenseOperator) = dense_identityoperator(op.basis_l, op.basis_r)


# Multiplication for Operators in terms of their gemv! implementation
function gemm!(alpha, M::Operator, b::DenseOperator, beta, result::DenseOperator)
    for i=1:size(b.data, 2)
        bket = Ket(b.basis_l, b.data[:,i])
        resultket = Ket(M.basis_l, result.data[:,i])
        gemv!(alpha, M, bket, beta, resultket)
        result.data[:,i] = resultket.data
    end
end

function gemm!(alpha, b::DenseOperator, M::Operator, beta, result::DenseOperator)
    for i=1:size(b.data, 1)
        bbra = Bra(b.basis_r, vec(b.data[i,:]))
        resultbra = Bra(M.basis_r, vec(result.data[i,:]))
        gemv!(alpha, bbra, M, beta, resultbra)
        result.data[i,:] = resultbra.data
    end
end

function *(op1::Operator, op2::DenseOperator)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = DenseOperator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op1::DenseOperator, op2::Operator)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = DenseOperator(op1.basis_l, op2.basis_r)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end

function *(op::Operator, psi::Ket)
    check_multiplicable(op.basis_r, psi.basis)
    result = Ket(op.basis_l)
    gemv!(Complex(1.), op, psi, Complex(0.), result)
    return result
end

function *(psi::Bra, op::Operator)
    check_multiplicable(psi.basis, op.basis_l)
    result = Bra(op.basis_r)
    gemv!(Complex(1.), psi, op, Complex(0.), result)
    return result
end

Base.prod{B<:Basis, T<:AbstractArray}(basis::B, operators::T) = (length(operators)==0 ? dense_identityoperator(basis) : prod(operators))

"""
Tensor product of operators where all missing indices are identity operators.

Arguments
---------
basis
    CompositeBasis of the resulting operator.
indices
    Indices of the subsystems in which the given operators live.
operators
    Operators defined in the subsystems.
"""
embed(basis::CompositeBasis, indices::Vector{Int}, operators::Vector) = tensor([prod(basis.bases[i], operators[find(indices.==i)]) for i=1:length(basis.bases)]...)
embed{T<:Operator}(basis::CompositeBasis, index::Int, op::T) = embed(basis, Int[index], T[op])


# Partial trace for dense operators.
function _strides(shape::Vector{Int})
    N = length(shape)
    S = zeros(Int, N)
    S[N] = 1
    for m=N-1:-1:1
        S[m] = S[m+1]*shape[m+1]
    end
    return S
end

@generated function _ptrace{RANK}(rank::Array{Int,RANK}, a::Matrix{Complex128},
                                  shape_l::Vector{Int}, shape_r::Vector{Int},
                                  indices::Vector{Int})
    return quote
        a_strides_l = _strides(shape_l)
        result_shape_l = deepcopy(shape_l)
        result_shape_l[indices] = 1
        result_strides_l = _strides(result_shape_l)
        a_strides_r = _strides(shape_r)
        result_shape_r = deepcopy(shape_r)
        result_shape_r[indices] = 1
        result_strides_r = _strides(result_shape_r)
        N_result_l = prod(result_shape_l)
        N_result_r = prod(result_shape_r)
        result = zeros(Complex128, N_result_l, N_result_r)
        @nexprs 1 (d->(Jr_{$RANK}=1;Ir_{$RANK}=1))
        @nloops $RANK ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{$RANK}=1;Il_{$RANK}=1))
            @nloops $RANK il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                #println("Jl_0: ", Jl_0, "; Jr_0: ", Jr_0, "; Il_0: ", Il_0, "; Ir_0: ", Ir_0)
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end


function check_ptrace_arguments(a::Operator, indices::Vector{Int})
    @assert length(a.basis_l.shape) == length(a.basis_r.shape)
    S = Set(indices)
    @assert length(S) == length(indices)
    @assert issubset(S, Set(1:length(a.basis_l.shape)))
end

"""
Partial trace of the given operator over the specified indices.
"""
function bases.ptrace(a::DenseOperator, indices::Vector{Int})
    check_ptrace_arguments(a, indices)
    if length(a.basis_l.shape) == length(indices)
        return trace(a)
    end
    rank = zeros(Int, [0 for i=1:length(a.basis_l.shape)]...)
    result = _ptrace(rank, a.data, a.basis_l.shape, a.basis_r.shape, indices)
    return DenseOperator(ptrace(a.basis_l, indices), ptrace(a.basis_r, indices), result)
end

bases.ptrace(a::Operator, index::Int) = bases.ptrace(a, Int[index])

"""
Partial trace of the given state vector over the specified indices.
"""
bases.ptrace(a::Ket, indices) = bases.ptrace(tensor(a, dagger(a)), indices)
bases.ptrace(a::Bra, indices) = bases.ptrace(tensor(dagger(a), a), indices)

"""
Change the ordering of the subsystems of the given operator.

Arguments
---------
rho
    A dense operator represented in a composite basis.
perm
    Vector defining the new ordering of the subsystems.
"""
function bases.permutesystems(rho::DenseOperator, perm::Vector{Int})
    @assert length(rho.basis_l.bases) == length(rho.basis_r.bases) == length(perm)
    @assert issubset(Set(1:length(rho.basis_l.bases)), Set(perm))
    data = reshape(rho.data, [reverse(rho.basis_l.shape); reverse(rho.basis_r.shape)]...)
    dataperm = length(perm) - reverse(perm) + 1
    data = permutedims(data, [dataperm; dataperm + length(perm)])
    data = reshape(data, length(rho.basis_l), length(rho.basis_r))
    DenseOperator(permutesystems(rho.basis_l, perm), permutesystems(rho.basis_r, perm), data)
end

end # module
