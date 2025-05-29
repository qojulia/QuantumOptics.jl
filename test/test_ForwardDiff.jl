using Test
using OrdinaryDiffEqCore, QuantumOptics
import ForwardDiff
import FiniteDiff

# for some caese ForwardDiff.jl returns NaN due to issue with DiffEq.jl. see https://github.com/SciML/DiffEqBase.jl/issues/861
# Here we test;
# That gradient from ForwardDiff.jl on QuantumOptics.jl match ForwardDiff.jl on DiffEq.jl.

# Note!
# gradient error is not directly related to the error of the state (abstol, reltol)
# partially related (here we use ForwardDiff and not some adjoint method) https://github.com/SciML/SciMLSensitivity.jl/issues/510
# here we partially control the gradient error by limiting step size (dtmax)


@testset "ForwardDiff on ODE Problems" begin

# schroedinger equation
b = SpinBasis(10//1)
psi0 = spindown(b)
H(p) = p[1]*sigmax(b) + p[2]*sigmam(b)
f_schrod!(dpsi, psi, p, t) = timeevolution.dschroedinger!(dpsi, H(p), psi)
function cost_schrod(p)
    prob = ODEProblem(f_schrod!, psi0, (0.0, pi), p)
    sol = solve(prob, DP5(); save_everystep=false)
    return 1 - norm(sol[end])
end

p = [rand(), rand()]
fordiff_schrod = ForwardDiff.gradient(cost_schrod, p)
findiff_schrod = FiniteDiff.finite_difference_gradient(cost_schrod, p)

@test isapprox(fordiff_schrod, findiff_schrod; atol=1e-2)

end

@testset "ForwardDiff with schroedinger" begin

# system
ba0 = FockBasis(2)
psi = basisstate(ba0, 1)
function getHt(p)
    op = [create(ba0)+destroy(ba0)]
    f(t) = sin(p*t)
    H_at_t = LazySum([f(0.0)], op)
    function Ht(t,_)
        H_at_t.factors .= (f(t),)
        return H_at_t
    end
    return Ht
end

# cost function
function cost(par, ψ0; kwargs...)
    opti = (;dtmax=exp2(-4), dt=exp2(-4))
    Ht = getHt(par)
    # this will rebuild the state with Dual elements
    _, ψT = timeevolution.schroedinger_dynamic((0.0, 1.0), ψ0, Ht; opti..., kwargs...)
    # this will not rebuild the state
    _, ψT = timeevolution.schroedinger((1.0, 2.0), last(ψT), Ht(0.5, ψ0); opti..., kwargs...)
    (abs2∘tr)( ψ0.data' * last(ψT).data ) # getting the data so this will work with also Bra states
end

# setup
p0 = rand()
δp = √eps()
# test
for u0 = (psi, psi', psi⊗psi') # test all methods of `rebuild`
    finite_diff_derivative = ( cost(p0+δp, u0) - cost(p0, u0) ) / δp
    Auto_diff_derivative = ForwardDiff.derivative(Base.Fix2(cost, u0), p0)
    @test isapprox(Auto_diff_derivative, finite_diff_derivative; atol=1e-5)
end

end # testset

@testset "ForwardDiff with schroedinger using TimeDependentSum" begin

base=SpinBasis(1/2)
ψi = spinup(base)
ψt = spindown(base)
function Ftdop(q)
    H = TimeDependentSum([q, abs2∘sinpi], [sigmaz(base), sigmax(base)])
    _, ψf = timeevolution.schroedinger_dynamic(range(0,1,2), ψi, H)
    abs2(ψt'last(ψf))
end
Ftdop(1.0)
@test ForwardDiff.derivative(Ftdop, 1.0) isa Any

function Ftdop(q)
    H = TimeDependentSum([1, abs2∘sinpi], [sigmaz(base), q*sigmax(base)])
    _, ψf = timeevolution.schroedinger_dynamic(range(0,1,2), ψi, H)
    abs2(ψt'last(ψf))
end
Ftdop(1.0)
@test ForwardDiff.derivative(Ftdop, 1.0) isa Any

end # testset


@testset "ForwardDiff with master" begin

b = SpinBasis(1//2)
psi0 = spindown(b)
rho0 = dm(psi0)
params = [rand(), rand()]

for f in (:(timeevolution.master), :(timeevolution.master_h), :(timeevolution.master_nh))
    # test to see if parameter propagates through Hamiltonian
    H(p) = p[1]*sigmax(b) + p[2]*sigmam(b)  # Hamiltonian
    function cost_H(p) #
        tf, psif = eval(f)((0.0, pi), rho0, H(p), [sigmax(b)])
        return 1 - norm(psif)
    end

    forwarddiff_H = ForwardDiff.gradient(cost_H, params)
    finitediff_H = FiniteDiff.finite_difference_gradient(cost_H, params)
    @test isapprox(forwarddiff_H, finitediff_H; atol=1e-2)

    # test to see if parameter propagates through Jump operator
    J(p) = p[1]*sigmax(b) + p[2]*sigmam(b)  # jump operator
    function cost_J(p)
        tf, psif = eval(f)((0.0, pi), rho0, sigmax(b), [J(p)])
        return 1 - norm(psif)
    end

    forwarddiff_J = ForwardDiff.gradient(cost_J, params)
    finitediff_J = FiniteDiff.finite_difference_gradient(cost_J, params)
    @test isapprox(forwarddiff_J, finitediff_J; atol=1e-2)
end

end
