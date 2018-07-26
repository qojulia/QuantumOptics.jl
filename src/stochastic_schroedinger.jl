module stochastic_schroedinger

export schroedinger, schroedinger_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...timeevolution
import ...timeevolution: integrate_stoch, recast!
import ...timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger

import DiffEqCallbacks

const DiffArray = Union{Vector{ComplexF64}, Array{ComplexF64, 2}}

"""
    stochastic.schroedinger(tspan, state0, H, Hs[; fout, ...])

Integrate stochastic Schrödinger equation.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial state as Ket.
* `H`: Deterministic part of the Hamiltonian.
* `Hs`: Stochastic part(s) of the Hamiltonian (either an operator or a vector
        of operators).
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `normalize_state=false`: Specify whether or not to normalize the state after
        each time step taken by the solver.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger(tspan, psi0::Ket, H::Operator, Hs::Vector;
                fout::Union{Function,Nothing}=nothing,
                normalize_state::Bool=false,
                calback=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)

    n = length(Hs)
    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    check_schroedinger(psi0, H)
    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger(psi, H, dpsi)
    dschroedinger_stoch(dx::DiffArray,
            t::Float64, psi::Ket, dpsi::Ket, n::Int) = dschroedinger_stochastic(dx, psi, Hs, dpsi, n)

    if normalize_state
        norm_func(u::Vector{ComplexF64}, t::Float64, integrator) = normalize!(u)
        ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
                 func_everystep=true,
                 func_start=false)
    else
        ncb = nothing
    end

    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n;
                    ncb=ncb,
                    kwargs...)
end
schroedinger(tspan, psi0::Ket, H::Operator, Hs::Operator; kwargs...) = schroedinger(tspan, psi0, H, [Hs]; kwargs...)

"""
    stochastic.schroedinger_dynamic(tspan, state0, fdeterm, fstoch[; fout, ...])

Integrate stochastic Schrödinger equation with dynamic Hamiltonian.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial state.
* `fdeterm`: Function `f(t, psi, u) -> H` returning the deterministic
        (time- or state-dependent) part of the Hamiltonian.
* `fstoch`: Function `f(t, psi, u, du) -> Hs` returning a vector that
        contains the stochastic terms of the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `noise_processes=0`: Number of distinct white-noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch`. If unset, the number is calculated automatically
        from the function output.
        NOTE: Set this number if you want to avoid an initial calculation of
        the function output!
* `normalize_state=false`: Specify whether or not to normalize the state after
        each time step taken by the solver.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, psi0::Ket, fdeterm::Function, fstoch::Function;
                fout::Union{Function,Nothing}=nothing, noise_processes::Int=0,
                normalize_state::Bool=false,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)

    if noise_processes == 0
        fs_out = fstoch(0.0, psi0)
        n = length(fs_out)
    else
        n = noise_processes
    end

    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger_dynamic(t, psi, fdeterm, dpsi)
    dschroedinger_stoch(dx::DiffArray,
            t::Float64, psi::Ket, dpsi::Ket, n::Int) =
        dschroedinger_stochastic(dx, t, psi, fstoch, dpsi, n)

    if normalize_state
        norm_func(u::Vector{ComplexF64}, t::Float64, integrator) = normalize!(u)
        ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
                 func_everystep=true,
                 func_start=false)
    else
        ncb = nothing
    end

    integrate_stoch(tspan, dschroedinger_determ, dschroedinger_stoch, x0, state,
            dstate, fout, n;
            ncb=ncb,
            kwargs...)
end


function dschroedinger_stochastic(dx::Vector{ComplexF64}, psi::Ket, Hs::Vector{T},
            dpsi::Ket, index::Int) where T <: Operator
    check_schroedinger(psi, Hs[index])
    recast!(dx, dpsi)
    dschroedinger(psi, Hs[index], dpsi)
end
function dschroedinger_stochastic(dx::Array{ComplexF64, 2}, psi::Ket, Hs::Vector{T},
            dpsi::Ket, n::Int) where T <: Operator
    for i=1:n
        check_schroedinger(psi, Hs[i])
        dx_i = @view dx[:, i]
        recast!(dx_i, dpsi)
        dschroedinger(psi, Hs[i], dpsi)
        recast!(dpsi, dx_i)
    end
end
function dschroedinger_stochastic(dx::DiffArray,
            t::Float64, psi::Ket, f::Function, dpsi::Ket, n::Int)
    ops = f(t, psi)
    dschroedinger_stochastic(dx, psi, ops, dpsi, n)
end

recast!(psi::StateVector, x::SubArray{ComplexF64, 1}) = (x .= psi.data)
recast!(x::SubArray{ComplexF64, 1}, psi::StateVector) = (psi.data = x)

end # module
