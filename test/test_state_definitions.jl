using Test
using QuantumOptics

@testset "state_definitions" begin

n=30
b=FockBasis(n)    
omega=40.3
T=2.3756
r=thermalstate(omega*number(b),T)
for k=1:n-1
    @test isapprox(r.data[k+1,k+1]/r.data[k,k],exp(-omega/T))
end
S=entropy_vn(r)
z=sum(exp.(-[0:n;]*omega))
s=expect(omega*number(b),r)/T+log(z)
isapprox(S,s)    

alpha=rand()+im*rand()
r=coherentthermalstate(b,omega*number(b),T,alpha)
@test isapprox(expect(number(b),r),abs(alpha)^2+1/(exp(omega/T)-1))
@test isapprox(expect(destroy(b),r),alpha)
@test isapprox(entropy_vn(r),S)

rp=phase_average(r)
@test isapprox(expect(number(b),rp),abs(alpha)^2+1/(exp(omega/T)-1)) 
@test isapprox(expect(destroy(b),rp),0)
for k=1:n
    @test isapprox(rp.data[k,k],r.data[k,k])
end

rpas=passive_state(r)
for k=1:n-1
    @test real(rpas.data[k+1,k+1])<real(rpas.data[k,k])
end
@test isapprox(expect(number(b),rpas),1/(exp(omega/T)-1))
@test isapprox(expect(destroy(b),rpas),0)
@test isapprox(entropy_vn(rpas),S)
 
end # testset
