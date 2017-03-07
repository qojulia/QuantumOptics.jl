using BenchmarkTools
using QuantumOptics

function coherentstate1(b::FockBasis, alpha::Number)
    alpha = complex(alpha)
    x = zeros(Complex128, b.Nmax - b.Nmin + 1)
    if b.Nmin == 0
        x[1] = exp(-abs2(alpha)/2)
    else
        x[1] = exp(-abs2(alpha)/2)*alpha^b.Nmin/sqrt(factorial(b.Nmin))
    end
    @inbounds for n=1:(b.Nmax-b.Nmin)
        x[n+1] = x[n]*alpha/sqrt(b.Nmin + n)
    end
    return Ket(b, x)
end

function coherentstate2(b::FockBasis, alpha::Number, result=Ket(b, Vector{Complex128}(b.shape[1])))
    alpha = complex(alpha)
    data = result.data
    if b.Nmin == 0
        data[1] = exp(-abs2(alpha)/2)
    else
        data[1] = exp(-abs2(alpha)/2)*alpha^b.Nmin/sqrt(factorial(b.Nmin))
    end
    @inbounds for n=1:(b.Nmax-b.Nmin)
        data[n+1] = data[n]*alpha/sqrt(b.Nmin + n)
    end
    return result
end

N = 10
b = FockBasis(N)
alpha = complex(1.)
tmp = Ket(b, Vector{Complex128}(b.shape[1]))
println(norm(coherentstate1(b, alpha) - coherentstate2(b, alpha)))

r1 = @benchmark coherentstate1($b, $alpha)
r2 = @benchmark coherentstate2($b, $alpha, $tmp)

println(r1)
println(r2)
