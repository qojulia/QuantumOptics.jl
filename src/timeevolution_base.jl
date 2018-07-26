using ..metrics

import OrdinaryDiffEq, DiffEqCallbacks, StochasticDiffEq

const DiffArray = Union{Vector{ComplexF64}, Array{ComplexF64, 2}}

function recast! end

"""
    integrate(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function; kwargs...)

Integrate using OrdinaryDiffEq
"""
function integrate(tspan::Vector{Float64}, df::Function, x0::DiffArray,
            state::T, dstate::T, fout::Function;
            alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm = OrdinaryDiffEq.DP5(),
            steady_state = false, tol = 1e-3, save_everystep = false,
            callback = nothing, kwargs...) where T

    function df_(dx::DiffArray, x::DiffArray, p, t)
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end
    function fout_(x::DiffArray, t::Float64, integrator)
        recast!(x, state)
        fout(t, state)
    end

    out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(state)})

    out = DiffEqCallbacks.SavedValues(Float64,out_type)

    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan,
                                         save_everystep=save_everystep,
                                         save_start = false)

    prob = OrdinaryDiffEq.ODEProblem{true}(df_, x0,(tspan[1],tspan[end]))

    if steady_state
        affect! = function (integrator)
            !save_everystep && scb.affect!(integrator,true)
            OrdinaryDiffEq.terminate!(integrator)
        end
        _cb = OrdinaryDiffEq.DiscreteCallback(
                                SteadyStateCondtion(copy(state),tol,state),
                                affect!;
                                save_positions = (false,false))
        cb = OrdinaryDiffEq.CallbackSet(_cb,scb)
    else
        cb = scb
    end

    full_cb = OrdinaryDiffEq.CallbackSet(callback,cb)

    sol = OrdinaryDiffEq.solve(
                prob,
                alg;
                reltol = 1.0e-6,
                abstol = 1.0e-8,
                save_everystep = false, save_start = false,
                save_end = false,
                callback=full_cb, kwargs...)
    out.t,out.saveval
end

function integrate(tspan::Vector{Float64}, df::Function, x0::DiffArray,
            state::T, dstate::T, ::Nothing; kwargs...) where T
    function fout(t::Float64, state::T)
        copy(state)
    end
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
end

struct SteadyStateCondtion{T,T2,T3}
    rho0::T
    tol::T2
    state::T3
end
function (c::SteadyStateCondtion)(rho,t,integrator)
    timeevolution.recast!(rho,c.state)
    dt = integrator.dt
    drho = metrics.tracedistance(c.rho0, c.state)
    c.rho0.data[:] = c.state.data
    drho/dt < c.tol
end



"""
    integrate_stoch(tspan::Vector{Float64}, df::Function, dg::Vector{Function}, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function; kwargs...)

Integrate using StochasticDiffEq
"""
function integrate_stoch(tspan::Vector{Float64}, df::Function, dg::Function, x0::Vector{ComplexF64},
            state::T, dstate::T, fout::Function, n::Int;
            save_everystep = false, callback=nothing,
            alg::StochasticDiffEq.StochasticDiffEqAlgorithm=StochasticDiffEq.EM(),
            noise_rate_prototype = nothing,
            noise_prototype_classical = nothing,
            noise=nothing,
            ncb=nothing,
            kwargs...) where T

    function df_(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, p, t)
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end

    function dg_(dx::Union{Vector{ComplexF64}, Array{ComplexF64, 2}},
                x::Vector{ComplexF64}, p, t)
        recast!(x, state)
        dg(dx, t, state, dstate, n)
    end

    function fout_(x::Vector{ComplexF64}, t::Float64, integrator)
        recast!(x, state)
        fout(t, state)
    end

    nc = isa(noise_prototype_classical, Nothing) ? 0 : size(noise_prototype_classical)[2]
    if isa(noise, Nothing) && n > 0
        noise_ = StochasticDiffEq.RealWienerProcess!(0.0, randn(n + nc))
    else
        noise_ = noise
    end
    if isa(noise_rate_prototype, Nothing)
        if n > 1 || nc > 1 || (n > 0 && nc > 0)
            noise_rate_prototype = zeros(ComplexF64, length(x0), n + nc)
        end
    end

    out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(state)})

    out = DiffEqCallbacks.SavedValues(Float64,out_type)

    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan,
                                         save_everystep=save_everystep,
                                         save_start = false)

    full_cb = OrdinaryDiffEq.CallbackSet(callback, ncb, scb)

    prob = StochasticDiffEq.SDEProblem{true}(df_, dg_, x0,(tspan[1],tspan[end]),
                    noise=noise_,
                    noise_rate_prototype=noise_rate_prototype)

    sol = StochasticDiffEq.solve(
                prob,
                alg;
                reltol = 1.0e-3,
                abstol = 1.0e-3,
                save_everystep = false, save_start = false,
                save_end = false,
                callback=full_cb, kwargs...)

    out.t,out.saveval
end

"""
    integrate_stoch

Define fout if it was omitted.
"""
function integrate_stoch(tspan::Vector{Float64}, df::Function, dg::Function, x0::Vector{ComplexF64},
    state::T, dstate::T, ::Nothing, n::Int; kwargs...) where T
    function fout(t::Float64, state::T)
        copy(state)
    end
    integrate_stoch(tspan, df, dg, x0, state, dstate, fout, n; kwargs...)
end


Base.@pure pure_inference(fout,T) = Core.Inference.return_type(fout, T)
