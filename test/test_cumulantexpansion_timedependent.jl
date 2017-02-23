using Base.Test
using QuantumOptics

@testset "cumulantexpansion_timedependent" begin

N = 3
Ncutoff = 2
T = [0.:0.1:1.;]

Ω = [0 2 3;
     2 0 1;
     3 1 0]

ω = [1., 1.2, 1.5]

basis_fock = FockBasis(Ncutoff)
basis = tensor([basis_fock for i=1:N]...)

a = destroy(basis_fock)
at = create(basis_fock)
I = identityoperator(basis_fock)


rho0 = cumulantexpansion.ProductDensityOperator([coherentstate(basis_fock, i%Ncutoff) for i=1:N]...)

J = LazyTensor[]
Jdagger = LazyTensor[]

# Interaction picture
Hrot = LazyTensor[]
for i=1:N, j=1:N
    if i==j
        continue
    end
    h = LazyTensor(basis, [i,j], [a, at], Ω[i,j])
    push!(Hrot, h)
end
Hrot = LazySum(Hrot...)

# Schroedinger picture
function f(t, rho)
    H = LazyTensor[LazyTensor(basis, i, at*a, ω[i]) for i=1:N]
    for i=1:N, j=1:N
        if i==j
            continue
        end
        h = LazyTensor(basis, [i,j], [a, at], exp(1im*(ω[i]-ω[j])*t)*Ω[i,j])
        push!(H, h)
    end
    LazySum(H...), J, Jdagger
end

tout, rho_rot_t = cumulantexpansion.master(T, rho0, Hrot, J)
tout, rho_t = cumulantexpansion.master_timedependent(T, rho0, f)

for (i, t) in enumerate(tout)
    R = prod([embed(basis, i, expm(1im*ω[i]*t*full(at*a))) for i=1:N])
    rho_rot = rho_rot_t[i]
    rho = rho_t[i]
    @test tracedistance(full(rho_rot), full(R)*full(rho)*dagger(full(R))) < 1e-5
end

end # testset
