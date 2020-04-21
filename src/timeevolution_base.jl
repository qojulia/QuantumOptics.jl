using QuantumOpticsBase
using QuantumOpticsBase: check_samebases, check_multiplicable

import OrdinaryDiffEq, DiffEqCallbacks

const DiffArray = Union{Vector{ComplexF64}, Array{ComplexF64, 2}}

function recast! end

"""
    integrate(tspan::Vector{Float64}, df::Function, x0::Vector{ComplexF64},
            state::T, dstate::T, fout::Function; kwargs...)

Integrate using OrdinaryDiffEq
"""
function integrate(tspan::Vector{Float64}, df::Function, x0::DiffArray,
            state::T, dstate::T, fout::Function;
            alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm = OrdinaryDiffEq.DP5(),
            steady_state = false, tol = 1e-3, save_everystep = false, saveat=tspan,
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

    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=saveat,
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
    drho = tracedistance(c.rho0, c.state)
    c.rho0.data[:] = c.state.data
    drho/dt < c.tol
end


const QO_CHECKS = Ref(true)
"""
    @skiptimechecks

Macro to skip checks during time-dependent problems.
Useful for `timeevolution.master_dynamic` and similar functions.
"""
macro skiptimechecks(ex)
    return quote
        QO_CHECKS.x = false
        local val = $(esc(ex))
        QO_CHECKS.x = true
        val
    end
end

Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)
