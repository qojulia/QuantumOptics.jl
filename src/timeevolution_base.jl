using ..ode_dopri

import OrdinaryDiffEq, DiffEqCallbacks

function recast! end

"""
df(t, state::T, dstate::T)
"""
function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function; callback = nothing, kwargs...)

    function df_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end
    function fout_(t::Float64, x::Vector{Complex128},integrator)
        recast!(x, state)
        fout(t, state)
    end

    # TODO: Infer the output of `fout` instead of computing it
    recast!(x0, state)
    out = DiffEqCallbacks.SavedValues(Float64,typeof(fout(tspan[1], state)))

    # Build callback solve with DP5
    # TODO: Expose algorithm choice
    cb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan)

    if callback == nothing
        _cb = cb
    else
        _cb = OrdinaryDiffEq.CallbackSet(cb,callback)
    end
    sol = OrdinaryDiffEq.solve(
                OrdinaryDiffEq.ODEProblem{true}(df_, x0,(tspan[1],tspan[end])),
                OrdinaryDiffEq.DP5();
                reltol = 1.0e-6,
                abstol = 1.0e-8,
                save_everystep = false, save_start = false, save_end = false,
                callback=_cb, kwargs...)
    out.t,out.saveval
end

function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, ::Void; kwargs...)
    function fout(t::Float64, state::T)
        copy(state)
    end
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
end
