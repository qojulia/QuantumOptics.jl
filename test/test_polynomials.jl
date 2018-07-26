using Test
using QuantumOptics
using QuantumOptics.polynomials


@testset "polynomials" begin

# Test Horner scheme
c = [0.2, 0.6, 1.7]
x0 = 1.3
@test horner(c, x0) == c[1] + c[2]*x0 + c[3]*x0^2

# Test Hermite polynomials
an = Vector{Vector{Int}}(8)
an[1] = [1]
an[2] = [0,2]
an[3] = [-2, 0, 4]
an[4] = [0, -12, 0, 8]
an[5] = [12, 0, -48, 0, 16]
an[6] = [0, 120, 0, -160, 0, 32]
an[7] = [-120, 0, 720, 0, -480, 0, 64]
an[8] = [0, -1680, 0, 3360, 0, -1344, 0, 128]
@test hermite.a(7) == an

A = hermite.A(7)
for n=0:7
    @test A[n+1] ≈ an[n+1]/sqrt(2^n*factorial(n))
end

end # testset
