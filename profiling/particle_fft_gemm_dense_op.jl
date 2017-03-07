using BenchmarkTools
using QuantumOptics

srand(0)

function gemm1(alpha::Complex128, A::DenseOperator, B::particle.FFTOperator, beta::Complex128, result::DenseOperator)
    if beta != Complex(0.)
        data = Matrix{Complex128}(size(result.data, 1), size(result.data, 2))
    else
        data = result.data
    end
    copy!(data, A.data)
    scale!(data, B.mul_after)
    conj!(data)
    B.fft_l2! * data
    conj!(data)
    scale!(data, B.mul_before)
    if alpha != Complex(1.)
        scale!(alpha, data)
    end
    if beta != Complex(0.)
        scale!(result.data, beta)
        result.data += data
    end
    nothing
end

xmin = -1.1
xmax = 2.5
Npoints = 10
b_pos = PositionBasis(xmin, xmax, Npoints)
b_mom = MomentumBasis(b_pos)

alpha = complex(2.)
beta = complex(3.)
op = DenseOperator(b_pos, rand(Complex128, Npoints, Npoints))
T = particle.FFTOperator(b_pos, b_mom)
result = DenseOperator(b_pos, b_mom, rand(Complex128, Npoints, Npoints))


result0 = (alpha*op*full(T) + beta*result)
y = alpha*(op*T) + beta*result
println(sum(abs(y.data - result0.data)))

result1 = copy(result)
gemm1(alpha, op, T, beta, result1)
println(sum(abs(result1.data - result0.data)))

result2 = copy(result)
operators.gemm!(alpha, op, T, beta, result2)
println(sum(abs(result2.data - result0.data)))



function run_gemm1(N, alpha, op, T, beta, result)
    for i=1:N
        gemm1(alpha, op, T, beta, result)
    end
end

# Profile.clear()
# @profile run_gemm1(1000, alpha, op, T, beta, result)

# r0 = @benchmark operators.gemm!($alpha, $op, $T, $beta, $result)
# r1 = @benchmark gemm1($alpha, $op, $T, $beta, $result)

# println(r0)
# println(r1)