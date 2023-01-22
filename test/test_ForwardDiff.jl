using Test
using OrdinaryDiffEq, QuantumOptics
import ForwardDiff

# for some caese ForwardDiff.jl returns NaN due to issue with DiffEq.jl. see https://github.com/SciML/DiffEqBase.jl/issues/861
# Here we test;
# That gradient from ForwardDiff.jl on QuantumOptics.jl match ForwardDiff.jl on DiffEq.jl.

# Note!
# gradient error is not directly related to the error of the state (abstol, reltol)
# partially related (here we use ForwardDiff and not some adjoint method) https://github.com/SciML/SciMLSensitivity.jl/issues/510
# here we partially control the gradient error by limiting step size (dtmax)


@testset "ForwardDiff with schroedinger" begin

# system
ba0 = FockBasis(2)
psi = basisstate(ba0, 1)
target0 = basisstate(ba0, 2)
function getHt(p)
    op = [create(ba0)+destroy(ba0)]
    f(t) = sin(p*t)
    H_at_t = LazySum([f(0)], op)
    function Ht(t,_)
        H_at_t.factors .= (f(t),)
        return H_at_t
    end
    return Ht
end

# cost function
function cost(par; kwargs...)
    opti = (;dtmax=exp2(-4), dt=exp2(-4))
    Ht = getHt(par)
    # this will rebuild the Bra with Dual elements
    _, ψT = timeevolution.schroedinger_dynamic((0.0, 0.2), psi'     , Ht; opti..., kwargs...)
    # this will not rebuild the Bra
    _, ψT = timeevolution.schroedinger_dynamic((0.2, 0.4), last(ψT) , Ht; opti..., kwargs...) 
    # this will not rebuild the Ket
    # also tests static schroedinger
    _, ψT = timeevolution.schroedinger((0.4, 0.6), last(ψT)', Ht(1.0, ψT); opti..., kwargs...)
    # this will not rebuild the Ket
    _, ψT = timeevolution.schroedinger_dynamic((0.6, 0.8), last(ψT)⊗last(ψT)', Ht; opti..., kwargs...)
    abs2(target0'*last(ψT)*target0)
end

# setup
p0 = rand()
δp = √eps()
# test
finite_diff_derivative = ( cost(p0+δp) - cost(p0) ) / δp
Auto_diff_derivative = ForwardDiff.derivative(cost, p0)
@test isapprox(Auto_diff_derivative, finite_diff_derivative; atol=1e-5)

end # testset
