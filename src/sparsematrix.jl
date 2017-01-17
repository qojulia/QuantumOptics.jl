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
