using BenchmarkTools
using QuantumOptics

function momentumoperator1(b::PositionBasis)
    b_mom = MomentumBasis(b)
    particle.FFTOperator(b, b_mom)*full(momentumoperator(b_mom))*particle.FFTOperator(b_mom, b)
end


# function momentumoperator2(b::PositionBasis)
#     x = particle.samplepoints(b)
#     dx = particle.spacing(b)
#     p = particle.samplepoints(MomentumBasis(b))
#     dp = particle.spacing(MomentumBasis(b))
#     data = Matrix{Complex128}(b.N, b.N)
#     for j=1:b.N, i=1:b.N
#         data[i,j] = complex(0.)
#         for k=1:b.N
#             data[i,j] += exp(1im*(p[i]-p[j])*x[k])*x[k]
#         end
#         data[i,j] *= dp^2/2pi
#     end
#     return DenseOperator(b, data)
# end

# function momentumoperator3(b::PositionBasis)
#     dx = particle.spacing(b)
#     dp = 2pi/(dx*b.N)
#     data = Matrix{Complex128}(b.N, b.N)
#     @inbounds for j=1:b.N, i=1:b.N
#         dp_ij = (j-i)*dp
#         data[i,j] = complex(0.)
#         @inbounds for k=1:b.N
#             xk = b.xmin + (k-1)*dx
#             data[i,j] += cis(-dp_ij*xk)*xk
#         end
#         data[i,j] *= dp^2/2pi
#     end
#     return DenseOperator(b, data)
# end

function momentumoperator4(b::PositionBasis)
    x = particle.samplepoints(b)
    dx = particle.spacing(b)
    b_mom = MomentumBasis(b)
    p = particle.samplepoints(b_mom)
    dp = particle.spacing(b_mom)
    data = cis(x*p.')
    scale!(p, data)
    scale!(cis(-b.xmin*(p-b_mom.pmin)), data)
    fft!(data, 1)
    scale!(cis(-b_mom.pmin*x), data)
    scale!(data, 1./b.N)
    return DenseOperator(b, data)
end

xmin = -1
xmax = 1
Npoints = 100

b_pos = PositionBasis(xmin, xmax, Npoints)
b_mom = MomentumBasis(b_pos)

p1 = momentumoperator1(b_pos)
p2 = momentumoperator2(b_pos)
p3 = momentumoperator3(b_pos)
p4 = momentumoperator4(b_pos)

println(sum(abs(p1.data-p2.data)))
println(sum(abs(p1.data-p3.data)))
println(sum(abs(p1.data-p4.data)))

# r1 = @benchmark momentumoperator1($b_pos)
# r2 = @benchmark momentumoperator2($b_pos)
# r3 = @benchmark momentumoperator3($b_pos)

# println(r1)
# println(r2)
# println(r3)
