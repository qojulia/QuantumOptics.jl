using QuantumOpticsBase
using QuantumOpticsBase: check_samebases, check_multiplicable

import OrdinaryDiffEq, DiffEqCallbacks, DiffEqBase, ForwardDiff

function recast! end

"""
    integrate(tspan, df::Function, x0::Vector{ComplexF64},
            state::T, dstate::T, fout::Function; kwargs...)

Integrate using OrdinaryDiffEq
"""
function integrate(tspan, df, x0,
            state, dstate, fout;
            alg = OrdinaryDiffEq.DP5(),
            steady_state = false, tol = 1e-3, save_everystep = false, saveat=tspan,
            callback = nothing, kwargs...)

    df_ = let df = df
        function df_(dx, x, p, t)
            recast!(state,x)
            recast!(dstate,dx)
            df(t, state, dstate)
            recast!(dx,dstate)
            return nothing
        end
    end

    fout_ = let fout = fout, state = state
        function fout_(x, t, integrator)
            recast!(state,x)
            fout(t, state)
        end
    end

    tType = float(eltype(tspan))
    out_type = pure_inference(fout, Tuple{tType,typeof(state)})

    out = DiffEqCallbacks.SavedValues(tType,out_type)

    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=saveat,
                                         save_everystep=save_everystep,
                                         save_start = false,
                                         tdir = first(tspan)<last(tspan) ? one(eltype(tspan)) : -one(eltype(tspan)))

    prob = OrdinaryDiffEq.ODEProblem{true}(df_, x0,(convert(tType, tspan[1]),convert(tType, tspan[end])))

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

function integrate(tspan, df, x0,
            state, dstate, ::Nothing; kwargs...)
    function fout(t, state)
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

function _check_const(op)
    if !QuantumOpticsBase.is_const(op)
        throw(
          ArgumentError("You are attempting to use a time-dependent dynamics generator " *
            "(a Hamiltonian or Lindbladian) with a solver that assumes constant " * 
            "dynamics. To avoid errors, please use the _dynamic solvers instead, " *
            "e.g. schroedinger_dynamic instead of schroedinger")
        )
    end
    nothing
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

function _promote_time_and_state(u0, H::AbstractOperator, tspan)
    Ts = eltype(H)
    Tt = real(Ts)
    p = Vector{Tt}(undef,0)
    u0_promote = DiffEqBase.promote_u0(u0, p, tspan[1])
    tspan_promote = DiffEqBase.promote_tspan(u0_promote.data, p, tspan, nothing, Dict{Symbol, Any}())
    return tspan_promote, u0_promote
end
function _promote_time_and_state(u0, H::AbstractOperator, J, tspan)
    Ts = DiffEqBase.promote_dual(eltype(H), DiffEqBase.anyeltypedual(J))
    Tt = real(Ts)
    p = Vector{Tt}(undef,0)
    u0_promote = DiffEqBase.promote_u0(u0, p, tspan[1])
    tspan_promote = DiffEqBase.promote_tspan(u0_promote.data, p, tspan, nothing, Dict{Symbol, Any}())
    return tspan_promote, u0_promote
end

_promote_time_and_state(u0, f, tspan) = _promote_time_and_state(u0, f(first(tspan)..., u0), tspan)

@inline function DiffEqBase.promote_u0(u0::Ket, p, t0)
    u0data_promote = DiffEqBase.promote_u0(u0.data, p, t0)
    if u0data_promote !== u0.data
        u0_promote = Ket(u0.basis, u0data_promote)
        return u0_promote
    end
    return u0
end
@inline function DiffEqBase.promote_u0(u0::Bra, p, t0)
    u0data_promote = DiffEqBase.promote_u0(u0.data, p, t0)
    if u0data_promote !== u0.data
        u0_promote = Bra(u0.basis, u0data_promote)
        return u0_promote
    end
    return u0
end
@inline function DiffEqBase.promote_u0(u0::Operator, p, t0)
    u0data_promote = DiffEqBase.promote_u0(u0.data, p, t0)
    if u0data_promote !== u0.data
        u0_promote = Operator(u0.basis_l, u0.basis_r, u0data_promote)
        return u0_promote
    end
    return u0
end