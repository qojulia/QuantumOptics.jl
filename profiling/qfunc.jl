using BenchmarkTools
using QuantumOptics

function qfunc1(rho::Operator, alpha::Complex128)
    psi = coherentstate(rho.basis_l, alpha)
    x = rho*psi
    a = dagger(psi)*x
    return real(a)/pi
end

function qfunc1(rho::Operator, X::Vector{Float64}, Y::Vector{Float64})
    @assert rho.basis_l == rho.basis_r
    return Float64[qfunc(rho, complex(x,y)) for x=X, y=Y]
end

function coherentstate2(b::FockBasis, alpha::Number, result=Ket(b, Vector{Complex128}(b.shape[1])))
    alpha = complex(alpha)
    data = result.data
    data[1] = exp(-abs2(alpha)/2)
    @inbounds for n=1:b.N
        data[n+1] = data[n]*alpha/sqrt(n)
    end
    return result
end

function qfunc2(rho::DenseOperator, alpha::Complex128,
                tmp1=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1])),
                tmp2=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))
                )
    # @assert rho.basis_l == rho.basis_r
    b = rho.basis_l
    coherentstate2(b, alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return real(a)/pi
end

function qfunc2(rho::Operator, X::Vector{Float64}, Y::Vector{Float64})
    @assert rho.basis_l == rho.basis_r
    b = rho.basis_l
    Nx = length(X)
    Ny = length(Y)
    tmp1 = Ket(b, Vector{Complex128}(b.shape[1]))
    tmp2 = Ket(b, Vector{Complex128}(b.shape[1]))
    result = Matrix{Float64}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        result[i, j] = qfunc2(rho, complex(X[i], Y[j]), tmp1, tmp2)
    end
    return result
end

N = 5
b = FockBasis(N)
rho = DenseOperator(b, rand(Complex128, N+1, N+1))
X = [-1:0.1:1;]
Y = [-1:0.1:1;]

alpha = complex(1.)
# println(qfunc1(rho, alpha))
# println(qfunc2(rho, alpha))

# qfunc1(rho, alpha)
# qfunc2(rho, alpha)

# qfunc1(rho, alpha)
# qfunc2(rho, alpha)

function run_qfunc1(N, rho, X, Y)
    for i in 1:N
        qfunc1(rho, X, Y)
    end
end

function run_qfunc2(N, rho, X, Y)
    for i in 1:N
        qfunc2(rho, X, Y)
    end
end

# @time run_qfunc1(1000, rho, alpha)
# @time run_qfunc1(100000, rho, alpha)

# @time run_qfunc2(1000, rho, alpha)
# @time run_qfunc2(100000, rho, alpha)


# Profile.clear()
# @profile run_qfunc1(1000, rho, X, Y)

# tmp1=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))
# tmp2=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))


r1 = @benchmark qfunc1($rho, $X, $Y)
r2 = @benchmark qfunc2($rho, $X, $Y)

println(r1)
println(r2)
