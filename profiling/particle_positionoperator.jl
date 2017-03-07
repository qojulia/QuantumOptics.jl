using BenchmarkTools
using QuantumOptics

function positionoperator1(b::MomentumBasis)
    b_pos = PositionBasis(b)
    particle.FFTOperator(b, b_pos)*full(positionoperator(b_pos))*particle.FFTOperator(b_pos, b)
end

function positionoperator2(b::MomentumBasis)
    p = particle.samplepoints(b)
    dp = particle.spacing(b)
    b_pos = PositionBasis(b)
    x = particle.samplepoints(b_pos)
    dx = particle.spacing(b_pos)
    data = Matrix{Complex128}(b.N, b.N)
    for j=1:b.N, i=1:b.N
        data[i,j] = complex(0.)
        for k=1:b.N
            data[i,j] += exp(-1im*(x[i]-x[j])*p[k])*p[k]
        end
        data[i,j] *= dx^2/2pi
    end
    return DenseOperator(b, data)
end

function positionoperator3(b::MomentumBasis)
    dp = particle.spacing(b)
    dx = 2pi/(dp*b.N)
    data = Matrix{Complex128}(b.N, b.N)
    for j=1:b.N, i=1:b.N
        dx_ij = (j-i)*dx
        data[i,j] = complex(0.)
        for k=1:b.N
            pk = b.pmin + (k-1)*dp
            data[i,j] += cis(dx_ij*pk)*pk
        end
        data[i,j] *= dx^2/2pi
    end
    return DenseOperator(b, data)
end

function positionoperator4(b::MomentumBasis)
    p = particle.samplepoints(b)
    dp = particle.spacing(b)
    b_pos = PositionBasis(b)
    x = particle.samplepoints(b_pos)
    dx = particle.spacing(b_pos)
    data = cis(p*x.')
    scale!(x, data)
    scale!(cis(-b.pmin*(x-b_pos.xmin)), data)
    fft!(data, 1)
    scale!(cis(-b_pos.xmin*p), data)
    scale!(data, 1./b.N)
    return DenseOperator(b, data)
end

xmin = -1
xmax = 1
Npoints = 300

b_pos = PositionBasis(xmin, xmax, Npoints)
b_mom = MomentumBasis(b_pos)

p1 = positionoperator1(b_mom)
p2 = positionoperator2(b_mom)
p3 = positionoperator3(b_mom)
p4 = positionoperator4(b_mom)

println(sum(abs(p1.data-p2.data)))
println(sum(abs(p1.data-p3.data)))
println(sum(abs(p1.data-p4.data)))

function run_f4(N, b_mom)
    for i=1:N
        positionoperator4(b_mom)
    end
end

# Profile.clear()
# @profile run_f4(1000, b_mom)


r1 = @benchmark positionoperator1($b_mom)
r2 = @benchmark positionoperator2($b_mom)
r3 = @benchmark positionoperator3($b_mom)
r4 = @benchmark positionoperator4($b_mom)

println(r1)
println(r2)
println(r3)
println(r4)