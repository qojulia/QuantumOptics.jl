using BenchmarkTools
using QuantumOptics

srand(0)

function gemm1(alpha::Complex128, A::particle.FFTOperator, B::DenseOperator, beta::Complex128, result::DenseOperator)
    if beta != Complex(0.)
        data = Matrix{Complex128}(size(result.data, 1), size(result.data, 2))
    else
        data = result.data
    end
    copy!(data, B.data)
    scale!(A.mul_before, data)
    A.fft_r2! * data
    scale!(A.mul_after, data)
    if alpha != Complex(1.)
        scale!(alpha, data)
    end
    if beta != Complex(0.)
        result.data += data
    end
    nothing
end

function gemm2(alpha::Complex128, A::particle.FFTOperator, B::DenseOperator, beta::Complex128, result::DenseOperator)
    @assert beta==Complex(0.)
    N = length(B.basis_l)
    M = length(B.basis_r)
    for j=1:N, i=1:M
        result.data[i,j] = A.mul_before[i]*B.data[i,j]
    end
    fft!(result.data, 1)
    for j=1:N, i=1:M
        result.data[i,j] *= A.mul_after[i]*alpha
    end
    nothing
end


xmin = -2
xmax = 2
Npoints = 100
b_pos = PositionBasis(xmin, xmax, Npoints)
b_mom = MomentumBasis(b_pos)
op = DenseOperator(b_pos, rand(Complex128, Npoints, Npoints))

result = DenseOperator(b_mom, b_pos, rand(Complex128, Npoints, Npoints))
T = particle.FFTOperator(b_mom, b_pos)


alpha = complex(1.)
beta = complex(1.)
result0 = (alpha*full(T)*op + beta*result)

gemm1(alpha, T, op, beta, result)
println(sum(abs(result.data - result0.data)))

# gemm2(alpha, T, op, beta, result)
# println(sum(abs(result.data - (full(T)*op).data)))

# println(sum(abs(result.data - result0)))

function run_gemm1(N, alpha, T, op, beta, result)
    for i=1:N
        gemm1(alpha, T, op, beta, result)
    end
end

Profile.clear()
@profile run_gemm1(1000, alpha, T, op, beta, result)

r0 = @benchmark operators.gemm!($alpha, $T, $op, $beta, $result)
r1 = @benchmark gemm1($alpha, $T, $op, $beta, $result)
# r2 = @benchmark gemm2($alpha, $T, $op, $beta, $result)


println(r0)
println(r1)
# println(r2)