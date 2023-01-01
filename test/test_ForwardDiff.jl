using Test
using OrdinaryDiffEq, QuantumOptics
touch("LocalPreferences.toml")
file = open("LocalPreferences.toml", "w")
write(file, """
[ForwardDiff]
nansafe_mode = true
"""
import ForwardDiff as FD

tests_repetition = 2^3

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
ba0 = FockBasis(5)
psi = basisstate(ba0, 1)
target0 = basisstate(ba0, 2)
#psi = randstate(ba0)
#target0 = randstate(ba0)
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

function cost01(par)
    Ht = getHt(par)
    ts = eltype(par).((0.0, 1.0))
    _, ψT = timeevolution.schroedinger_dynamic(ts, psi, Ht; dtmax=exp2(-4))
    abs2(target0'*last(ψT))
end

cost01(rand())
FD.derivative(cost01, rand())
@test all([isapprox(FD.derivative(cost01, q), (cost01(q+ε)-cost01(q))/ε, atol=1e-7) for q=range(0,2π,tests_repetition)])

## static
function get_H(p)
    op = create(ba0)+destroy(ba0)
    return sin(p)*op
end

function cost02(par)
    # test the promotion switch -> not to get Dual(Complex(Dual{...}))
    Ts = promote_type(eltype(par), eltype(psi))
    PSI = Ket(psi.basis, Ts.(psi.data))
    H = get_H(par)
    ts = (0.0, 1.0)
    _, ψT = timeevolution.schroedinger(ts, PSI, H; dtmax=exp2(-4)) # using dtmax here to improve derivative accuracy, specifically for par=0
    abs2(target0'*last(ψT))
end

cost02(rand())
FD.derivative(cost02, rand())
@test all([isapprox(FD.derivative(cost02, q), (cost02(q+ε)-cost02(q))/ε, atol=1e-7) for q=range(0,2π,tests_repetition)])

# ex1
## 3 level kerr transmon with drive
ba1 = FockBasis(2)
T2 = (1+rand())*1e4
ω00, α0 = 0.1randn(), -0.2+0.05rand()
function get_Ht(p::Vector{<:Tp}) where Tp
    ω0, α = Tp(ω00), Tp(α0)
    A, freq, ϕ, T = p
    op = 2π*([number(ba1), 2\create(ba1)*number(ba1)*destroy(ba1), im*(create(ba1)-destroy(ba1))])
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
ψ01 = Operator(SpinBasis(1/2), basisstate(ba1, 1), basisstate(ba1, 2))
## target states
target1 = ψ01*exp(im*0.5π*dense(sigmax(SpinBasis(1/2)))) # x gate
## cost function using QO.jl
function cost1(par)
    T = par[4]
    Ht = get_Ht(par)
    ts = (0.0, T)
    _, ψT = timeevolution.schroedinger_dynamic(ts, ψ01, Ht; abstol=1e-9, reltol=1e-9, dtmax=exp2(-5))
    1-abs2(tr(target1'last(ψT))/2)*exp(-T/T2)
end

p0 = [0.3, ω00, 0.25, 10.0]
rp(k) = p0 .* ( ones(length(p0))+k*(1 .-2rand(length(p0))) )

cost1(p0)
FD.gradient(cost1, p0)
@test all([isapprox(FD.gradient(cost1, p), fin_diff(cost1, p); atol=1e-6) for p=rp.(range(0.0, 0.1, tests_repetition))])

# ex2
ba2 = FockBasis(3)
A, B = randoperator(ba2), randoperator(ba2)
A+=A'
B+=B'
ψ02 = randstate(ba2)
target2 = randstate(ba2)
function cost2(par)
    a,b = par
    Ht(t,u) = A + a*cos(b*t)*B/10
    _, ψT = timeevolution.schroedinger_dynamic((0.0, 1.0, 2.0), ψ02, Ht; abstol=1e-9, reltol=1e-9, dtmax=0.05)
    abs2(tr(target2'ψT[2])) + abs2(tr(ψ02'ψT[3]))
end

cost2(rand(2))
FD.gradient(cost2, rand(2))
@test all([begin
    p = randn(2)
    g1 = FD.gradient(cost2, p)
    g2 = fin_diff(cost2, p)
    isapprox(g1, g2 ; atol=1e-6)
    end for _=1:tests_repetition])


end # testset
