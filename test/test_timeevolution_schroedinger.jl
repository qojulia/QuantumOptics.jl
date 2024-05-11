using Test
using QuantumOptics

@testset "schroedinger" begin

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


psi0 = tensor([coherentstate(basis_fock, i%Ncutoff) for i=1:N]...)


# Interaction picture
Hrot = SparseOpType[]
for i=1:N, j=1:N
    if i==j
        continue
    end
    h = embed(basis, [i,j], [a, Ω[i,j]*at])
    push!(Hrot, h)
end
Hrot = sum(Hrot)

# Schroedinger picture
function f(t, psi)
    H = SparseOpType[embed(basis, i, ω[i]*at*a) for i=1:N]
    for i=1:N, j=1:N
        if i==j
            continue
        end
        h = embed(basis, [i,j], [a, exp(1im*(ω[i]-ω[j])*t)*Ω[i,j]*at])
        push!(H, h)
    end
    sum(H)
end

tout, psi_rot_t = timeevolution.schroedinger(T, psi0, Hrot)
tout, psi_t = timeevolution.schroedinger_dynamic(T, psi0, f)

n_op = dense(at*a)
for (i, t) in enumerate(tout)
    R = prod([embed(basis, i, exp(1im*ω[i]*t*n_op)) for i=1:N])
    psi_rot = psi_rot_t[i]
    psi = psi_t[i]
    # @test abs(dagger(psi_rot)*R*psi) < 1e-5
    rho = dm(psi)
    rho_rot = dm(psi_rot)
    @test tracedistance(rho_rot, dense(R)*rho*dagger(dense(R))) < 1e-5
end

function fout(t, psi)
 deepcopy(psi)
end
t_fout, psi_fout = timeevolution.schroedinger(T, psi0, Hrot; fout=fout)
@test t_fout == tout && psi_fout == psi_rot_t


# propagate few different states separately
subspace = basisstate.([basis], (Ncutoff+1).^(0:N-1).+1)
t_list, psi_list = timeevolution.schroedinger_dynamic.([T], subspace, f; abstol=eps(), reltol=eps()) |> q->(getindex.(q,1), getindex.(q,2))
# propagate the same states as before, together as an projection from SubSpaceBasis
proj1 = projector(basis, SubspaceBasis(basis, subspace))
t_sub1, psi_sub1 = timeevolution.schroedinger_dynamic(T, proj1, f; abstol=eps(), reltol=eps())
# propagate the same states as before, together as a transformation from NLevelBasis
proj2 = Operator(proj1.basis_l, NLevelBasis(size(proj1.data,2)), proj1.data)
t_sub2, psi_sub2 = timeevolution.schroedinger_dynamic(T, proj2, f; abstol=eps(), reltol=eps())

# check that time vector is the same
@test t_list[1:end-1] == t_list[2:end] && t_list[1] == t_sub1 == t_sub2
# check that base is preserved
@test all(getfield.(psi_sub1, :basis_l) .== [proj1.basis_l]) && all(getfield.(psi_sub1, :basis_r) .== [proj1.basis_r])
@test all(getfield.(psi_sub2, :basis_l) .== [proj2.basis_l]) && all(getfield.(psi_sub2, :basis_r) .== [proj2.basis_r])
# check that data is independent of basis_r
@test all(getfield.(psi_sub1, :data) .≈ getfield.(psi_sub2, :data))
# check that data is the same
@test all([hcat(q...) for q=eachrow(getfield.(hcat(psi_list...),:data))] .≈ getfield.(psi_sub1, :data) .≈ getfield.(psi_sub2, :data))
## same for schroedinger
t_list, psi_list = timeevolution.schroedinger.([T], subspace, f.(0.0, subspace); abstol=eps(), reltol=eps()) |> q->(getindex.(q,1), getindex.(q,2))
t_sub1, psi_sub1 = timeevolution.schroedinger(T, proj1, f(0.0, proj1); abstol=eps(), reltol=eps())
t_sub2, psi_sub2 = timeevolution.schroedinger(T, proj2, f(0.0, proj2); abstol=eps(), reltol=eps())
@test t_list[1:end-1] == t_list[2:end] && t_list[1] == t_sub1 == t_sub2 # check that time vector is the same
@test all(getfield.(psi_sub1, :basis_l) .== [proj1.basis_l]) && all(getfield.(psi_sub1, :basis_r) .== [proj1.basis_r]) # check that base is preserved
@test all(getfield.(psi_sub2, :basis_l) .== [proj2.basis_l]) && all(getfield.(psi_sub2, :basis_r) .== [proj2.basis_r])
@test all(getfield.(psi_sub1, :data) .≈ getfield.(psi_sub2, :data)) # check that data is independent of basis_r
@test all([hcat(q...) for q=eachrow(getfield.(hcat(psi_list...),:data))] .≈ getfield.(psi_sub1, :data) .≈ getfield.(psi_sub2, :data)) # check that data is the same

# test integration of propagator using 2 level system
basis = SpinBasis(1//2)
su = spinup(basis)
u0 = dense(identityoperator(basis))
sx = sigmax(basis)
sz = sigmaz(basis)

# for the time dependent equation
f(t, psi) = sx * π
tspan = 0:1.0
t, u = timeevolution.schroedinger(tspan, u0, π * sx)

# I think the tolerance on the differential equation is 1e-6, we expect the operator to be essentially the identity
@test abs(expect(sz, u[end] * su)) - abs(expect(sz, u0 * su)) < 1e-6

t, u = timeevolution.schroedinger_dynamic(tspan, u0, f)
@test abs(expect(sz, u[end] * su)) - abs(expect(sz, u0 * su)) < 1e-6



end # testset


@testset "reverse time schroedinger" begin

# time span
tl0 = 3rand()-1.5 |> q -> range(q-0.5, q+0.5, 2^7)
# basis
bas = GenericBasis(rand(2:4))
# states to propagate
ψ0 = Operator([randstate(bas) for _=1:3])
# Hamiltonian 
op_list = [randoperator(bas) for _=1:4]
op_list.+= dagger.(op_list)
fun_list = [cos, sin, abs2, exp]
Ht(t,_) = sum(f(t)*op for (f,op)=zip(fun_list, op_list))
# propagate
tol=1e-12
## moving forward
tl  , ψl  = timeevolution.schroedinger_dynamic(tl0         ,      ψ0,  Ht;  abstol=tol, reltol=tol);
## propagate final state backwards
tlr , ψlr = timeevolution.schroedinger_dynamic(reverse(tl0), last(ψl), Ht;  abstol=tol, reltol=tol);

# test reverse output time is indeed reverse
@test tlr == reverse(tl0)
# test that the state is traced back to the initial state
@test all(isapprox.(ψl, reverse(ψlr), rtol=100tol))

end # testset
