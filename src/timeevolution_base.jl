using ..ode_dopri

function recast! end

"""
df(t, state::T, dstate::T)
"""
function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, fout::Function; kwargs...)
    function df_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        recast!(x, state)
        recast!(dx, dstate)
        df(t, state, dstate)
        recast!(dstate, dx)
    end
    function fout_(t::Float64, x::Vector{Complex128})
        recast!(x, state)
        fout(t, state)
    end
    ode(df_, tspan, x0, fout_; kwargs...)
end

function integrate{T}(tspan::Vector{Float64}, df::Function, x0::Vector{Complex128},
            state::T, dstate::T, ::Void; kwargs...)
    tout = Float64[]
    xout = T[]
    function fout(t::Float64, state::T)
        push!(tout, t)
        push!(xout, copy(state))
    end
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
    (tout, xout)
end