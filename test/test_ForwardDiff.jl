using Test
using OrdinaryDiffEq, QuantumOptics
import ForwardDiff as FD

function fin_diff(fun, x::Vector, ind::Int; ϵ)
    dx = zeros(length(x))
    dx[ind]+= ϵ/2
    ( fun(x+dx) - fun(x-dx) ) / ϵ
end

fin_diff(fun, x; ϵ=√eps()) = [fin_diff(fun, x, k; ϵ) for k=1:length(x)]

@testset "ForwardDiff with schroedinger" begin

# ex0
ε = √eps()
## dynamic
ba = FockBasis(5)
ψ0 = basisstate(ba, 1)
target = basisstate(ba, 2)
#ψ0 = randstate(ba)
#target = randstate(ba)
function getHt(p)
    op = [create(ba)+destroy(ba)]
    f(t) = sin(p*t)
    H_at_t = LazySum([f(0)], op)
    function Ht(t,_)
        H_at_t.factors .= (f(t),)
        return H_at_t
    end
    return Ht
end

function cost01(par)
    Ht = getHt(par)
    ts = (0.0, 1.0)
    _, ψT = timeevolution.schroedinger_dynamic(ts, ψ0, Ht)
    abs2(target'*last(ψT))
end

@test all([isapprox(FD.derivative(cost01, 1.0), (cost01(1.0+ε)-cost01(1.0))/ε, atol=1e-7) for q=range(0,2π,2^7)])

## static
function get_H(p)
    op = create(ba)+destroy(ba)
    return sin(p)*op
end

function cost02(par)
    H = get_H(par)
    ts = (0.0, 1.0)
    _, ψT = timeevolution.schroedinger(ts, ψ0, H)
    abs2(target'*last(ψT))
end

@test all([isapprox(FD.derivative(cost02, 1.0), (cost02(1.0+ε)-cost02(1.0))/ε, atol=1e-7) for q=range(0,2π,2^7)])

# ex1
## 3 level kerr transmon with drive
ba = FockBasis(2)
T2 = (1+rand())*1e4
ω00, α0 = 0.1randn(), -0.2+0.05rand()
function get_Ht(p::Vector{<:Tp}) where Tp
    ω0, α = Tp(ω00), Tp(α0)
    A, freq, ϕ, T = p
    op = 2π*([number(ba), 2\create(ba)*number(ba)*destroy(ba), im*(create(ba)-destroy(ba))])
    fω(t) = ω0
    fα(t) = α
    fΩ(t) = A*cospi(2t*freq + 2ϕ)*sinpi(t/T)^2
    H_at_t = LazySum(zeros(Tp,length(op)), op)
    function Ht(t,_)
        H_at_t.factors.= (fω(t), fα(t), fΩ(t))
        return H_at_t
    end
    return Ht
end
## initial states
ψ0 = Operator(SpinBasis(1/2), basisstate(ba, 1), basisstate(ba, 2))
## target states
target = ψ0*exp(im*0.5π*dense(sigmax(SpinBasis(1/2)))) # x gate
## cost function using QO.jl
function cost(par)
    T = par[4]
    Ht = get_Ht(par)
    ts = (0.0, T)
    _, ψT = timeevolution.schroedinger_dynamic(ts, ψ0, Ht; abstol=1e-9, reltol=1e-9, alg=nothing, dtmax=1e-2)
    1-abs2(tr(target'last(ψT))/2)*exp(-T/T2)
end

p0 = [0.03, ω00, 0.25, 100.0]
rp(k) = p0 .* ( ones(length(p0))+k*(1 .-2rand(length(p0))) )

@test all([isapprox(FD.gradient(cost, p), fin_diff(cost, p); atol=1e-6) for p=rp.(range(0.0, 0.1, 2^7))])

# ex2
ba = FockBasis(3)
A, B = randoperator(ba), randoperator(ba)
A+=A'
B+=B'
ψ02 = randstate(ba)
target2 = randstate(ba)
function cost2(par)
    a,b = par
    Ht(t,u) = A + a*cos(b*t)*B/10
    _, ψT = timeevolution.schroedinger_dynamic((0.0, 1.0, 2.0), ψ02, Ht; abstol=1e-9, reltol=1e-9, alg=nothing, dtmax=0.05)
    abs2(tr(target2'ψT[2])) + abs2(tr(ψ02'ψT[3]))
end

@test all([begin
    p = randn(2)
    g1 = FD.gradient(cost2, p)
    g2 = fin_diff(cost2, p)
    isapprox(g1, g2 ; atol=1e-6)
    end for _=1:2^7])


end # testset