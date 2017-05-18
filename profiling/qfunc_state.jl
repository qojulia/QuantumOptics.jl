using BenchmarkTools
using QuantumOptics

function _qfunc_ket(x::Vector{Complex128}, conj_alpha::Complex128)
    s = x[1]
    @inbounds for i=2:length(x)
        s = x[i] + s*conj_alpha
    end
    abs2(s)*exp(-abs2(conj_alpha))/pi
end

function qfunc1(psi::Ket, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(psi)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    N = length(b)::Int
    x = similar(psi.data)
    x[N] = psi.data[1]
    n = 1.
    @inbounds for i in 1:N-1
        x[N-i] = psi.data[i+1]/n
        n *= sqrt(i+1)
    end
    result = Matrix{Float64}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        _conj_alpha = complex(xvec[i], -yvec[j])/sqrt(2)
        result[i, j] = _qfunc_ket(x, _conj_alpha)
    end
    return result
end

function qfunc2(psi::Ket, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(psi)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    points = Nx*Ny
    N = length(b)::Int
    _conj_alpha = [complex(x, -y)/sqrt(2) for x=xvec, y=yvec]
    q = fill(psi.data[N]/sqrt(N-1), size(_conj_alpha))
    @inbounds for n=1:N-2
        f0_ = 1/sqrt(N-n-1)
        x = psi.data[N-n]
        for i=1:points
            q[i] = (x + q[i]*_conj_alpha[i])*f0_
        end
    end
    result = similar(q, Float64)
    x = psi.data[1]
    @inbounds for i=1:points
        result[i] = abs2(x + q[i]*_conj_alpha[i])*exp(-abs2(_conj_alpha[i]))/pi
    end
    result
end


N = 200
b = FockBasis(N)
psi = coherentstate(b, 0.7+0.1im)
X = collect(linspace(-1, 1, 100))
Y = collect(linspace(-2, 1, 100))

# @code_warntype qfunc1(psi, X, Y)

q = qfunc1(psi, X, Y)
println(q[1])
q = qfunc2(psi, X, Y)
println(q[1])
println(qfunc(psi, -1, -2))

@time qfunc1(psi, X, Y)
@time qfunc1(psi, X, Y)
@time qfunc2(psi, X, Y)
@time qfunc2(psi, X, Y)

# function run_qfunc1(N, rho, X, Y)
#     for i in 1:N
#         qfunc1(rho, X, Y)
#     end
# end

# function run_qfunc2(N, rho, X, Y)
#     for i in 1:N
#         qfunc2(rho, X, Y)
#     end
# end

# @time run_qfunc1(1000, rho, alpha)
# @time run_qfunc1(100000, rho, alpha)

# @time run_qfunc2(1000, rho, alpha)
# @time run_qfunc2(100000, rho, alpha)


# Profile.clear()
# @profile run_qfunc1(1000, rho, X, Y)

# tmp1=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))
# tmp2=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1]))


r1 = @benchmark qfunc1($psi, $X, $Y)
r2 = @benchmark qfunc2($psi, $X, $Y)

println(r1)
println(r2)
