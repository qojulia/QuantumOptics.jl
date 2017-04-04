using Base.Test
using QuantumOptics

@testset "odedopri" begin

ode_dopri = QuantumOptics.ode_dopri
ode_event = ode_dopri.ode_event
ode = ode_dopri.ode

ζ = 0.5
ω₀ = 10.0

y₀ = Float64[0., sqrt(1-ζ^2)*ω₀]
A = 1
ϕ = 0


function f(t::Float64)
    α = sqrt(1-ζ^2)*ω₀
    x = A*exp(-ζ*ω₀*t)*sin(α*t + ϕ)
    p = A*exp(-ζ*ω₀*t)*(-ζ*ω₀*sin(α*t + ϕ) + α*cos(α*t + ϕ))
    return [x,p]
end

function df(t::Float64, y::Vector{Float64}, dy::Vector{Float64})
    dy[1] = y[2]
    dy[2] = -2*ζ*ω₀*y[2] - ω₀^2*y[1]
    return nothing
end


# Test absolute and relative tolerances
T = [0.:0.5:5.;]

for reltol=[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    tout, y = ode(df, T, y₀, reltol=reltol, abstol=0.)
    for i=2:length(T)
        # println("T = $(T[i])")
        # println("Target Reltol: $(reltol*T[i]); Real reltol: ", norm(y[i]-f(T[i]))/norm(y[i]))
        @test T[i]*reltol*20 > norm(y[i]-f(T[i]))/norm(y[i]) > T[i]*reltol/20
    end
end

for abstol=[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
    tout, y = ode(df, T, y₀, reltol=0., abstol=abstol)
    for i=2:length(T)
        # println("T = $(T[i])")
        # println("Target abstol: $(abstol*T[i]); Real abstol: ", norm(y[i]-f(T[i])))
        @test abstol*10 > norm(y[i]-f(T[i])) > abstol/100
    end
end


# Test output of intermediate steps
T = [0.,10.]
tout, yout = ode(df, T, y₀; display_intermediatesteps=true)
@test length(tout)>2
maxstep = maximum(tout[2:end]-tout[1:end-1])


# Test hmax stepszie control
hmax = maxstep/10
tout, yout = ode(df, T, y₀; display_intermediatesteps=true, hmax=hmax)
maxstep2 = maximum(tout[2:end]-tout[1:end-1])
@test (maxstep2-hmax)<1e-12


# Test hmin
@test_throws ErrorException ode(df, T, y₀; display_intermediatesteps=true, hmin=0.4)


# Test h0
h0 = 1e-5
tout, yout = ode(df, T, y₀; display_intermediatesteps=true, h0=h0)
@test abs(h0-(tout[2]-tout[1]))<1e-14
h0 = 0.4
tout, yout = ode(df, T, y₀; display_intermediatesteps=true, h0=h0)
@test abs(h0-(tout[2]-tout[1]))>0.3

# Test fout
tout_ = Float64[]
yout_ = Vector{Float64}[]
function fout(t, y)
    push!(tout_, t)
    push!(yout_, deepcopy(y))
end
T = [0.,0.5,1.0]
result = ode(df, T, y₀, fout)
@test result == nothing
tout, yout = ode(df, T, y₀)

@test tout == tout_
@test yout == yout_


# Test ode_event
T = [0, 0.4, 0.8, 1]
event_locator(t, y) = t - 0.5

tout, yout = ode_event(df, T, y₀,
            (t, y) -> t-0.5,
            (t, y) -> ode_dopri.stop)
@test tout[end] == 0.4
@test 1e-5 > norm(f(0.4) - yout[end])

tout, yout = ode_event(df, T, y₀,
            (t, y) -> t-0.5,
            (t, y) -> ode_dopri.nojump)
@test tout[end] == 1.
@test 1e-5 > norm(f(1.)-yout[end])

end # testset