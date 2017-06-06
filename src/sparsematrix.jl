module sparsematrix

const SparseMatrix = SparseMatrixCSC{Complex128, Int}


function gemm_sp_dense_small(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, result::Matrix{Complex128})
    if alpha == Complex128(1.)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = M.nzval[i]
                @inbounds for j=1:size(B, 2)
                    result[row, j] += val*B[colindex, j]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                row = M.rowval[i]
                val = alpha*M.nzval[i]
                @inbounds for j=1:size(B, 2)
                    result[row, j] += val*B[colindex, j]
                end
            end
        end
    end
end

function gemm_sp_dense_big(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, result::Matrix{Complex128})
    if alpha == Complex128(1.)
        @inbounds for j=1:size(B, 2)
            @inbounds for colindex = 1:M.n
                m2 = B[colindex, j]
                @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                    row = M.rowval[i]
                    result[row, j] += M.nzval[i]*m2
                end
            end
        end
    else
        @inbounds for j=1:size(B, 2)
            @inbounds for colindex = 1:M.n
                m2 = alpha*B[colindex, j]
                @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                    row = M.rowval[i]
                    result[row, j] += M.nzval[i]*m2
                end
            end
        end
    end
end

function gemm!(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
    if nnz(M) > 1000
        gemm_sp_dense_big(alpha, M, B, result)
    else
        gemm_sp_dense_small(alpha, M, B, result)
    end
end

function gemm!(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrix, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
    dimB = size(result,1)
    if alpha == Complex128(1.)
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = M.nzval[i]
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[j, colindex] += mi*B[j, mrowvali]
                end
            end
        end
    else
        @inbounds for colindex = 1:M.n
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                mi = M.nzval[i]*alpha
                mrowvali = M.rowval[i]
                @inbounds for j=1:dimB
                    result[j, colindex] += mi*B[j, mrowvali]
                end
            end
        end
    end
end

function gemv!(alpha::Complex128, M::SparseMatrix, v::Vector{Complex128}, beta::Complex128, result::Vector{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
    if alpha == Complex128(1.)
        @inbounds for colindex = 1:M.n
            vj = v[colindex]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i]] += M.nzval[i]*vj
            end
        end
    else
        @inbounds for colindex = 1:M.n
            vj = alpha*v[colindex]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i]] += M.nzval[i]*vj
            end
        end
    end
end

function gemv!(alpha::Complex128, v::Vector{Complex128}, M::SparseMatrix, beta::Complex128, result::Vector{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
    if alpha == Complex128(1.)
        @inbounds for colindex=1:M.n
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[colindex] += M.nzval[i]*v[M.rowval[i]]
            end
        end
    else
        @inbounds for colindex=1:M.n
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[colindex] += M.nzval[i]*alpha*v[M.rowval[i]]
            end
        end
    end
end

function sub2sub{N, M}(shape1::NTuple{N, Int}, shape2::NTuple{M, Int}, I::CartesianIndex{N})
    linearindex = sub2ind(shape1, I.I...)
    CartesianIndex(ind2sub(shape2, linearindex)...)
end

function ptrace(x, shape_nd::Vector{Int}, indices::Vector{Int})
    shape_nd = (shape_nd...)
    N = div(length(shape_nd), 2)
    shape_2d = (x.m, x.n)
    shape_nd_after = ([i ∈ indices || i-N ∈ indices ? 1 : shape_nd[i] for i=1:2*N]...)
    shape_2d_after = (prod(shape_nd_after[1:N]), prod(shape_nd_after[N+1:end]))
    I_nd_after_max = CartesianIndex(shape_nd_after...)
    y = spzeros(Complex128, shape_2d_after...)
    for I in eachindex(x)
        I_nd = sub2sub(shape_2d, shape_nd, I)
        if I_nd.I[indices] != I_nd.I[indices + N]
            continue
        end
        I_after = sub2sub(shape_nd_after, shape_2d_after, min(I_nd, I_nd_after_max))
        y[I_after] += x[I]
    end
    y
end

function permutedims(x, shape, perm)
    shape = (shape...)
    shape_perm = ([shape[i] for i in perm]...)
    y = spzeros(Complex128, x.m, x.n)
    for I in eachindex(x)
        linear_index = sub2ind((x.m, x.n), I.I...)
        cartesian_index = ind2sub(shape, linear_index)
        cartesian_index_perm = [cartesian_index[i] for i=perm]
        linear_index_perm = sub2ind(shape_perm, cartesian_index_perm...)
        J = ind2sub((x.m, x.n), linear_index_perm)
        y[J...] = x[I.I...]
    end
    y
end

end # module
