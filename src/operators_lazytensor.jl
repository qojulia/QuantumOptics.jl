"""
Lazy implementation of a tensor product of operators.

The suboperators are stored as values in a dictionary where the key is
the index of the subsystem. Additionally a complex factor is stored in the
"factor" field which allows for fast multiplication with a number.
"""
type LazyTensor <: Operator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    indices::Vector{Int}
    operators::Vector{Operator}

    function LazyTensor(op::LazyTensor, factor::Number)
        new(op.basis_l, op.basis_r, factor, op.indices, op.operators)
    end

    function LazyTensor(basis_l::Basis, basis_r::Basis,
                        indices::Vector{Int}, ops::Vector,
                        factor::Number=1)
        if typeof(basis_l) != CompositeBasis
            basis_l = CompositeBasis(basis_l.shape, Basis[basis_l])
        end
        if typeof(basis_r) != CompositeBasis
            basis_r = CompositeBasis(basis_r.shape, Basis[basis_r])
        end
        N = length(basis_l.bases)
        @assert N==length(basis_r.bases)
        sortedindices.check_indices(N, indices)
        @assert length(indices) == length(ops)
        for n=1:length(indices)
            @assert isa(ops[n], Operator)
            @assert ops[n].basis_l == basis_l.bases[indices[n]]
            @assert ops[n].basis_r == basis_r.bases[indices[n]]
        end
        if !issorted(indices)
            perm = sortperm(indices)
            indices = indices[perm]
            ops = ops[perm]
        end
        new(basis_l, basis_r, complex(factor), indices, ops)
    end
end

LazyTensor(basis::Basis, indices::Vector{Int}, ops::Vector, factor::Number=1) = LazyTensor(basis, basis, indices, ops, factor)
LazyTensor(basis_l::Basis, basis_r::Basis, index::Int, operator::Operator, factor::Number=1) = LazyTensor(basis_l, basis_r, [index], Operator[operator], factor)
LazyTensor(basis::Basis, index::Int, operators::Operator, factor::Number=1.) = LazyTensor(basis, basis, index, operators, factor)

suboperator(op::LazyTensor, i::Int) = op.operators[findfirst(op.indices, i)]
suboperators(op::LazyTensor, indices::Vector{Int}) = op.operators[[findfirst(op.indices, i) for i in indices]]

Base.full(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, DenseOperator[full(x) for x in op.operators])
Base.sparse(op::LazyTensor) = op.factor*embed(op.basis_l, op.basis_r, op.indices, SparseOperator[sparse(x) for x in op.operators])

==(x::LazyTensor, y::LazyTensor) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor==y.factor


function *(a::LazyTensor, b::LazyTensor)
    check_multiplicable(a.basis_r, b.basis_l)
    indices = sortedindices.union(a.indices, b.indices)
    ops = Vector{Operator}(length(indices))
    for n in 1:length(indices)
        i = indices[n]
        in_a = i in a.indices
        in_b = i in b.indices
        if in_a && in_b
            ops[n] = suboperator(a, i)*suboperator(b, i)
        elseif in_a
            ops[n] = suboperator(a, i)
        elseif in_b
            ops[n] = suboperator(b, i)
        end
    end
    return LazyTensor(a.basis_l, b.basis_r, indices, ops, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b, a*b.factor)
function *(a::LazyTensor, b::DenseOperator)
    operators.check_multiplicable(a.basis_r, b.basis_l)
    result = DenseOperator(a.basis_l, b.basis_r)
    operators.gemm!(complex(1.), a, b, complex(1.), result)
    result
end
function *(a::DenseOperator, b::LazyTensor)
    operators.check_multiplicable(a.basis_r, b.basis_l)
    result = DenseOperator(a.basis_l, b.basis_r)
    operators.gemm!(complex(1.), a, b, complex(1.), result)
    result
end

/(a::LazyTensor, b::Number) = LazyTensor(a, a.factor/b)

-(a::LazyTensor) = LazyTensor(a, -a.factor)

identityoperator(::Type{LazyTensor}, b1::Basis, b2::Basis) = LazyTensor(b1, b2, Int[], Operator[])


dagger(op::LazyTensor) = LazyTensor(op.basis_r, op.basis_l, op.indices, Operator[dagger(x) for x in op.operators], conj(op.factor))

function trace(op::LazyTensor)
    result = op.factor
    for i in 1:length(op.basis_l.bases)
        if i in op.indices
            result *= trace(suboperator(op, i))
        else
            result *= _identitylength(op, i)
        end
    end
    result
end

_identitylength(op::LazyTensor, i::Int) = min(length(op.basis_l.bases[i]), length(op.basis_r.bases[i]))

function ptrace(op::LazyTensor, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    N = length(op.basis_l.shape)
    rank = N - length(indices)
    if rank==0
        return trace(op)
    end
    factor = op.factor
    for i in indices
        if i in op.indices
            factor *= trace(suboperator(op, i))
        else
            factor *= _identitylength(op, i)
        end
    end

    remaining_indices = sortedindices.remove(op.indices, indices)
    if rank==1 && length(remaining_indices)==1
        return factor * suboperator(op, remaining_indices[1])
    end
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    if rank==1
        return factor * identityoperator(b_l, b_r)
    end
    ops = Vector{Operator}(length(remaining_indices))
    for i in 1:length(ops)
        ops[i] = suboperator(op, remaining_indices[i])
    end
    LazyTensor(b_l, b_r, sortedindices.shiftremove(op.indices, indices), ops, factor)
end

tensor(a::LazyTensor, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [a.indices; b.indices+length(a.basis_l.bases)], Operator[a.operators; b.operators], a.factor*b.factor)
tensor(a::LazyWrapper, b::LazyWrapper) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [1, 2], Operator[a.operator, b.operator], a.factor*b.factor)
tensor(a::LazyTensor, b::LazyWrapper) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [a.indices; length(a.basis_l.bases)+1], [a.operators; b.operator], a.factor*b.factor)
tensor(a::LazyWrapper, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, [1; b.indices], [a.operator; b.operators], a.factor*b.factor)

function permutesystems(op::LazyTensor, perm::Vector{Int})
    b_l = permutesystems(op.basis_l, perm)
    b_r = permutesystems(op.basis_r, perm)
    indices = [findfirst(perm, i) for i in op.indices]
    perm_ = sortperm(indices)
    LazyTensor(b_l, b_r, indices[perm_], op.operators[perm_], op.factor)
end

"""
Recursively calculate result_{IK} = \\sum_J op_{IJ} h_{JK}
"""
function _gemm_recursive_dense_lazy(i_k::Int, N_k::Int, K::Int, J::Int, val::Complex128,
                        shape::Vector{Int}, strides_k::Vector{Int}, strides_j::Vector{Int},
                        indices::Vector{Int}, h::LazyTensor,
                        op::Matrix{Complex128}, result::Matrix{Complex128})
    if i_k > N_k
        for I=1:size(op, 1)
            result[I, K] += val*op[I, J]
        end
        return nothing
    end
    if i_k in indices
        h_i = operators_lazy.suboperator(h, i_k)
        if isa(h_i, SparseOperator)
            h_i_data = h_i.data::SparseMatrixCSC{Complex128,Int}
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_dense_lazy(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DenseOperator)
            h_i_data = h_i.data::Matrix{Complex128}
            Nk = size(h_i_data, 2)
            Nj = size(h_i_data, 1)
            @inbounds for k=1:Nk
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for j=1:Nj
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data[j,k]
                    _gemm_recursive_dense_lazy(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        else
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    else
        @inbounds for k=1:shape[i_k]
            K_ = K + strides_k[i_k]*(k-1)
            J_ = J + strides_j[i_k]*(k-1)
            _gemm_recursive_dense_lazy(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
        end
    end
end

"""
Recursively calculate result_{JI} = \\sum_K h_{JK} op_{KI}
"""
function _gemm_recursive_lazy_dense(i_k::Int, N_k::Int, K::Int, J::Int, val::Complex128,
                        shape::Vector{Int}, strides_k::Vector{Int}, strides_j::Vector{Int},
                        indices::Vector{Int}, h::LazyTensor,
                        op::Matrix{Complex128}, result::Matrix{Complex128})
    if i_k > N_k
        for I=1:size(op, 2)
            result[J, I] += val*op[K, I]
        end
        return nothing
    end
    if i_k in indices
        h_i = operators_lazy.suboperator(h, i_k)
        if isa(h_i, SparseOperator)
            h_i_data = h_i.data::SparseMatrixCSC{Complex128,Int}
            @inbounds for k=1:h_i_data.n
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for jptr=h_i_data.colptr[k]:h_i_data.colptr[k+1]-1
                    j = h_i_data.rowval[jptr]
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data.nzval[jptr]
                    _gemm_recursive_lazy_dense(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        elseif isa(h_i, DenseOperator)
            h_i_data = h_i.data::Matrix{Complex128}
            Nk = size(h_i_data, 2)
            Nj = size(h_i_data, 1)
            @inbounds for k=1:Nk
                K_ = K + strides_k[i_k]*(k-1)
                @inbounds for j=1:Nj
                    J_ = J + strides_j[i_k]*(j-1)
                    val_ = val*h_i_data[j,k]
                    _gemm_recursive_lazy_dense(i_k+1, N_k, K_, J_, val_, shape, strides_k, strides_j, indices, h, op, result)
                end
            end
        else
            throw(ArgumentError("gemm! of LazyTensor is not implemented for $(typeof(h_i))"))
        end
    else
        @inbounds for k=1:shape[i_k]
            K_ = K + strides_k[i_k]*(k-1)
            J_ = J + strides_j[i_k]*(k-1)
            _gemm_recursive_lazy_dense(i_k + 1, N_k, K_, J_, val, shape, strides_k, strides_j, indices, h, op, result)
        end
    end
end

function gemm(alpha::Complex128, op::Matrix{Complex128}, h::LazyTensor, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(beta, result)
    end
    N_k = length(h.basis_r.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = operators_dense._strides(h.basis_l.shape)
    strides_k = operators_dense._strides(h.basis_r.shape)
    _gemm_recursive_dense_lazy(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

function gemm(alpha::Complex128, h::LazyTensor, op::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(beta, result)
    end
    N_k = length(h.basis_l.bases)
    shape = [min(h.basis_l.shape[i], h.basis_r.shape[i]) for i=1:length(h.basis_l.shape)]
    strides_j = operators_dense._strides(h.basis_l.shape)
    strides_k = operators_dense._strides(h.basis_r.shape)
    _gemm_recursive_lazy_dense(1, N_k, 1, 1, alpha*h.factor, shape, strides_k, strides_j, h.indices, h, op, result)
end

operators.gemm!(alpha::Complex128, h::LazyTensor, op::DenseOperator, beta::Complex128, result::DenseOperator) = gemm(alpha, h, op.data, beta, result.data)
operators.gemm!(alpha::Complex128, op::DenseOperator, h::LazyTensor, beta::Complex128, result::DenseOperator) = gemm(alpha, op.data, h, beta, result.data)

function operators.gemv!(alpha, a::LazyTensor, b::Ket, beta, result::Ket)
    b_data = reshape(b.data, length(b.data), 1)
    result_data = reshape(result.data, length(result.data), 1)
    gemm(alpha, a, b_data, beta, result_data)
end

function operators.gemv!(alpha, a::Bra, b::LazyTensor, beta, result::Bra)
    a_data = reshape(a.data, 1, length(a.data))
    result_data = reshape(result.data, 1, length(result.data))
    gemm(alpha, a_data, b, beta, result_data)
end
