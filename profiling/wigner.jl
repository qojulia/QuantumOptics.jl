using BenchmarkTools
using QuantumOptics

function f1(rho::DenseOperator, x::Vector{Float64}, y::Vector{Float64})
  b = basis(rho)
  @assert typeof(b) == FockBasis

  X = x./sqrt(2) # Normalization of alpha
  Y = y./sqrt(2)
  if abs2(maximum(abs(x)) + 1.0im*maximum(abs(y))) > 0.75*b.N
    warn("x and y range close to cut-off!")
  end

  W = zeros(Float64, length(x), length(y))
  @inbounds for i=1:length(x), j=1:length(y)
    alpha = (X[i] + 1.0im*Y[j])
    D = displace(b, alpha)
    op = dagger(D)*rho*D
    W[i, j] = real(sum([(-1)^k*op.data[k+1, k+1] for k=0:b.N]))
  end

  return W./pi
end

function f2(rho::DenseOperator, x::Vector{Float64}, y::Vector{Float64})
    b = basis(rho)
    @assert typeof(b) == FockBasis

    if abs2(maximum(abs(x)) + 1.0im*maximum(abs(y))) > 0.75*b.N
        warn("x and y range close to cut-off!")
    end

    W = Matrix{Float64}(length(x), length(y))
    @inbounds for i=1:length(x), j=1:length(y)
        alpha = complex(x[i], y[j])/sqrt(2)
        D = displace(b, alpha)
        op = dagger(D)*rho*D
        w = 0.
        for k=0:b.N
            w += (-1)^k*real(op.data[k+1, k+1])
        end
        W[i, j] = w/pi
    end
    return W
end


function run_f1(N, rho, x, y)
    for i in 1:N
        f1(rho, x, y)
    end
end

function run_f2(N, rho, x, y)
    for i in 1:N
        f2(rho, x, y)
    end
end


b = FockBasis(50)
rho = dm(coherentstate(b, 3+1im))
xvec = [-3:1.4:3;]

println("Difference 1,2: ", abs2(f1(rho, xvec, xvec) - f2(rho, xvec, xvec)))


Profile.clear()
@profile run_f2(20, rho, xvec, xvec)

# r0 = @benchmark wigner($rho, $xvec, $xvec) seconds=3
# r1 = @benchmark f1($rho, $xvec, $xvec) seconds=3
# r2 = @benchmark f2($rho, $xvec, $xvec) seconds=3

# println(r0)
# println(r1)
# println(r2)
