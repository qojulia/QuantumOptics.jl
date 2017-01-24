using Base.Cartesian

N1 = 23
N2 = 17
N3 = 19

srand(1)
x = rand(Complex128, N1, N2, N3, N1, N2, N3)


function ptrace_forloops(x)
    n1, n2, n3 = size(x)
    y = zeros(Complex128, n2, n3, n2, n3)
    for i5=1:n3
        for i4=1:n2
            for i3=1:n3
                for i2=1:n2
                    for i1=1:n1
                        y[i2,i3,i4,i5] += x[i1,i2,i3,i1,i4,i5]
                    end
                end
            end
        end
    end
    y
end

function ptrace_slicing(x::Array{Complex128, 6})
    n1, n2, n3 = size(x)
    y = zeros(Complex128, n2, n3, n2, n3)
    for i1=1:n1
        y += x[i1,:,:,i1,:,:]
    end
    y
end

function ptrace_cartesian(x::Array{Complex128, 6})
    n1, n2, n3 = size(x)
    y = zeros(Complex128, 1, n2, n3, 1, n2, n3)
    ymax = CartesianIndex(size(y))
    for I in CartesianRange(size(x))
        if I.I[1] != I.I[4]
            continue
        end
        y[min(ymax, I)] += x[I]
    end
    reshape(y, n2, n3, n2, n3)
end

function ptrace_cartesian2(x::Array{Complex128, 6})
    n1, n2, n3 = size(x)
    y = zeros(Complex128, 1, n2, n3, 1, n2, n3)
    for I in CartesianRange(size(y))
        for k in CartesianRange((n1, 1, 1))
            delta = CartesianIndex(k, k)
            y[I] += x[I+delta-1]
        end
    end
    reshape(y, n2, n3, n2, n3)
end

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

@generated function _ptrace(a::Matrix{Complex128},
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
        @nexprs 1 (d->(Jr_{3}=1;Ir_{3}=1))
        @nloops 3 ir (d->1:shape_r[d]) (d->(Ir_{d-1}=Ir_d; Jr_{d-1}=Jr_d)) (d->(Ir_d+=a_strides_r[d]; if !(d in indices) Jr_d+=result_strides_r[d] end)) begin
            @nexprs 1 (d->(Jl_{3}=1;Il_{3}=1))
            @nloops 3 il (k->1:shape_l[k]) (k->(Il_{k-1}=Il_k; Jl_{k-1}=Jl_k; if (k in indices && il_k!=ir_k) Il_k+=a_strides_l[k]; continue end)) (k->(Il_k+=a_strides_l[k]; if !(k in indices) Jl_k+=result_strides_l[k] end)) begin
                #println("Jl_0: ", Jl_0, "; Jr_0: ", Jr_0, "; Il_0: ", Il_0, "; Ir_0: ", Ir_0)
                result[Jl_0, Jr_0] += a[Il_0, Ir_0]
            end
        end
        return result
    end
end

function ptrace_nloop(x)
    n1, n2, n3 = size(x)
    n = n1*n2*n3
    x = reshape(x, n, n)
    y = _ptrace(x, [n3,n2,n1], [n3,n2,n1], [3])
    reshape(y, n2, n3, n2, n3)
end

dist(x,y) = sum(abs(x-y))
result = ptrace_forloops(x)

println(dist(result, ptrace_slicing(x)))
println(dist(result, ptrace_cartesian(x)))
println(dist(result, ptrace_cartesian2(x)))
println(dist(result, ptrace_nloop(x)))


println("Explicit loops")
@time ptrace_forloops(x)
@time ptrace_forloops(x)

println("Slicing")
@time ptrace_slicing(x)
@time ptrace_slicing(x)

println("Cartesian Index")
@time ptrace_cartesian(x)
@time ptrace_cartesian(x)

println("Cartesian Index 2")
@time ptrace_cartesian2(x)
@time ptrace_cartesian2(x)

println("nloop")
@time ptrace_nloop(x)
@time ptrace_nloop(x)
