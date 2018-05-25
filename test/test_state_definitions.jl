using Base.Test
using QuantumOptics

@testset "state_definitions" begin

n=20
c=.1+rand()
T=2.3756*rand()
r=thermalstate(c*number(FockBasis(n)),T)
for k=1:n-1
    @test isapprox(r.data[k+1,k+1]/r.data[k,k],exp(-c/T))
end

end # testset
