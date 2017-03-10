using BenchmarkTools
using Base.LinAlg
srand(0)

typealias SparseMatrix SparseMatrixCSC{Complex128}


function f1(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrix, beta::Complex128, result::Matrix{Complex128})
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
end

function f2(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrix, beta::Complex128, result::Matrix{Complex128})
    scale!(result, beta)
    dimB = size(result,1)
    @inbounds for colindex = 1:M.n
        @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            mi = M.nzval[i]*alpha
            mrowvali = M.rowval[i]
            @inbounds for j=1:dimB
                result[j,colindex] += mi*B[j,mrowvali]
            end
        end
    end
end

function f3(alpha::Complex128, B::Matrix{Complex128}, M::SparseMatrix, beta::Complex128, result::Matrix{Complex128})
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


N1 = 100
N2 = 120
N3 = 70

s = 0.01
alpha = complex(1.)
beta = complex(1.)

A = rand(Complex128, N3, N2)
B = sprand(Complex128, N2, N1, s)

result_ = rand(Complex128, N3, N1)

result1 = deepcopy(result_)
f1(alpha, A, B, beta, result1)
result2 = deepcopy(result_)
f2(alpha, A, B, beta, result2)
result3 = deepcopy(result_)
f3(alpha, A, B, beta, result3)

println(norm(result1-result2))
println(norm(result1-result3))

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

println(r1)
println(r2)
println(r3)
