module sparsematrix

typealias SparseMatrix SparseMatrixCSC{Complex128}


function gemm!(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:M.m
        for colindex = 1:M.n
            m2 = alpha*B[colindex, j]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i], j] += M.nzval[i]*m2
            end
        end
    end
    nothing
end

function gemm!(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrix, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            mi = M.nzval[i]*alpha
            mrowvali = M.rowval[i]
            for j=1:size(result,1)
                result[j,colindex] += mi*B[j,mrowvali]
            end
        end
    end
    nothing
end

function gemv!(alpha::Complex128, M::SparseMatrix, v::Vector{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n
        vj = alpha*v[colindex]
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[M.rowval[i]] += M.nzval[i]*vj
        end
    end
    nothing
end

function gemv!(alpha::Complex128, v::Vector{Complex128}, M::SparseMatrixCSC{Complex128}, beta::Complex128, result::Vector{Complex128})
    @inbounds for i=1:length(result)
        result[i] *= beta
    end
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            result[colindex] += M.nzval[i]*alpha*v[M.rowval[i]]
        end
    end
    nothing
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
