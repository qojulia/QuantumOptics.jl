using BenchmarkTools
using Base.LinAlg
srand(0)

typealias SparseMatrix SparseMatrixCSC{Complex128}


function f1(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    @inbounds for j=1:size(result,2), i=1:size(result,1)
        result[i,j] *= beta
    end
    @inbounds for j=1:size(B, 2)
        for colindex = 1:M.n
            m2 = alpha*B[colindex, j]
            for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i], j] += M.nzval[i]*m2
            end
        end
    end
    nothing
end

function f2(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    scale!(result, beta)
    @inbounds for j=1:size(B, 2)
        @inbounds for colindex = 1:M.n
            m2 = alpha*B[colindex, j]
            @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
                result[M.rowval[i], j] += M.nzval[i]*m2
            end
        end
    end
end

function f3(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
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

function f4(alpha::Complex128, M::SparseMatrix, B::Matrix{Complex128}, beta::Complex128, result::Matrix{Complex128})
    if beta == Complex128(0.)
        fill!(result, beta)
    elseif beta != Complex128(1.)
        scale!(result, beta)
    end
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

N1 = 50
N2 = 50
N3 = 70

s = 0.1
alpha = complex(1.)
beta = complex(1.)

A = sprand(Complex128, N1, N2, s)
B = rand(Complex128, N2, N3)

result_ = rand(Complex128, N1, N3)

result1 = deepcopy(result_)
f1(alpha, A, B, beta, result1)
result2 = deepcopy(result_)
f2(alpha, A, B, beta, result2)
result3 = deepcopy(result_)
f3(alpha, A, B, beta, result3)
result4 = deepcopy(result_)
f4(alpha, A, B, beta, result4)

println(norm(result1-result2))
println(norm(result1-result3))
println(norm(result1-result4))

function run_f1(N::Int, alpha, A, B, beta, result)
    for i=1:N
        f1(alpha, A, B, beta, result)
    end
end

function run_f2(N::Int, alpha, A, B, beta, result)
    for i=1:N
        f2(alpha, A, B, beta, result)
    end
end

function run_f3(N::Int, alpha, A, B, beta, result)
    for i=1:N
        f3(alpha, A, B, beta, result)
    end
end


# @time f1(alpha, A, B, beta, result1)
# @time f1(alpha, A, B, beta, result1)
# @time f2(alpha, A, B, beta, result2)
# @time f2(alpha, A, B, beta, result2)

Profile.clear()
# @profile run_f1(10000, alpha, A, B, beta, result1)
# @profile run_f2(10000, alpha, A, B, beta, result2)
# @profile run_f3(10000, alpha, A, B, beta, result3)

r1 = @benchmark f1($alpha, $A, $B, $beta, $result1)
r2 = @benchmark f2($alpha, $A, $B, $beta, $result2)
r3 = @benchmark f3($alpha, $A, $B, $beta, $result3)
r4 = @benchmark f4($alpha, $A, $B, $beta, $result4)

println(r1)
println(r2)
println(r3)
println(r4)
