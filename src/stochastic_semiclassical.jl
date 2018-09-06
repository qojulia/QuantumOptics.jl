module stochastic_semiclassical

export schroedinger_semiclassical, master_semiclassical

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...semiclassical
import ...semiclassical: recast!, State, dmaster_h_dynamic
using ...timeevolution
import ...timeevolution: integrate_stoch, QO_CHECKS
import ...timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger
import ...stochastic.stochastic_master: check_master_stoch
using ...stochastic
using LinearAlgebra

import DiffEqCallbacks

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}
const DiffArray = Union{Vector{ComplexF64}, Array{ComplexF64, 2}}

"""
    semiclassical.schroedinger_stochastic(tspan, state0, fquantum, fclassical[; fout, ...])

Integrate time-dependent Schrödinger equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `state0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, psi, u) -> H` returning the time and or state
        dependent Hamiltonian.
* `fclassical`: Function `f(t, psi, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the vector `du`.
* `fstoch_quantum=nothing`: Function `f(t, psi, u) -> Hs` that returns a vector
        of operators corresponding to the stochastic terms of the Hamiltonian.
        NOTE: Either this function or `fstoch_classical` has to be defined.
* `fstoch_classical=nothing`: Function `f(t, psi, u, du)` that calculates the
        stochastic terms of the derivative `du`.
        NOTE: Either this function or `fstoch_quantum` has to be defined.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `noise_processes=0`: Number of distinct quantum noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch`. If unset, the number is calculated automatically
        from the function output.
        NOTE: Set this number if you want to avoid an initial calculation of
        the function output!
* `noise_prototype_classical=nothing`: The equivalent of the optional argument
        `noise_rate_prototype` in `StochasticDiffEq` for the classical
        stochastic function `fstoch_classical` only. Must be set for
        non-diagonal classical noise or combinations of quantum and classical
        noise. See the documentation for details.
* `normalize_state=false`: Specify whether or not to normalize the state after
        each time step taken by the solver.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_semiclassical(tspan, state0::State{T}, fquantum::Function,
                fclassical::Function; fstoch_quantum::Union{Nothing, Function}=nothing,
                fstoch_classical::Union{Nothing, Function}=nothing,
                fout::Union{Function,Nothing}=nothing,
                noise_processes::Int=0,
                noise_prototype_classical=nothing,
                normalize_state::Bool=false,
                kwargs...) where T<:Ket
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_det(t::Float64, state::State{T}, dstate::State{T}) where T<:Ket =
            semiclassical.dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)

    if isa(fstoch_quantum, Nothing) && isa(fstoch_classical, Nothing)
        throw(ArgumentError("No stochastic functions provided!"))
    end

    x0 = Vector{ComplexF64}(undef, length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)

    if noise_processes == 0
        n = 0
        if isa(fstoch_quantum, Function)
            fs_out = fstoch_quantum(0.0, state0.quantum, state0.classical)
            n += length(fs_out)
        end
    else
        n = noise_processes
    end

    if n > 0 && isa(fstoch_classical, Function)
        if isa(noise_prototype_classical, Nothing)
            throw(ArgumentError("noise_prototype_classical must be set for combinations of quantum and classical noise!"))
        end
    end

    if normalize_state
        len_q = length(state0.quantum)
        function norm_func(u::Vector{ComplexF64}, t::Float64, integrator)
            u .= [normalize!(u[1:len_q]); u[len_q+1:end]]
        end
        ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
                 func_everystep=true,
                 func_start=false)
    else
        ncb = nothing
    end

    dschroedinger_stoch(dx::DiffArray, t::Float64, state::State{T}, dstate::State{T}, n::Int) where T<:Ket =
            dschroedinger_stochastic(dx, t, state, fstoch_quantum, fstoch_classical, dstate, n)

    integrate_stoch(tspan_, dschroedinger_det, dschroedinger_stoch, x0, state, dstate, fout, n;
                    noise_prototype_classical = noise_prototype_classical,
                    ncb=ncb,
                    kwargs...)
end

"""
    stochastic.master_semiclassical(tspan, rho0, H, Hs, J; <keyword arguments>)

Time-evolution according to a stochastic master equation.

For dense arguments the `master` function calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `fquantum`: Function `f(t, rho, u) -> (H, J, Jdagger)` or
        `f(t, rho, u) -> (H, J, Jdagger, rates)` giving the deterministic
        part of the master equation.
* `fclassical`: Function `f(t, rho, u, du)` that calculates the classical
        derivatives `du`.
* `fstoch_quantum=nothing`: Function `f(t, rho, u) -> C, Cdagger`
        that returns the stochastic operator for the superoperator of the form
        `C[i]*rho + rho*Cdagger[i]`.
* `fstoch_classical=nothing`: Function `f(t, rho, u, du)` that calculates the
        stochastic terms of the derivative `du`.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `noise_processes=0`: Number of distinct quantum noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch`. If unset, the number is calculated automatically
        from the function output.
        NOTE: Set this number if you want to avoid an initial calculation of
        the function output!
* `noise_prototype_classical=nothing`: The equivalent of the optional argument
        `noise_rate_prototype` in `StochasticDiffEq` for the classical
        stochastic function `fstoch_classical` only. Must be set for
        non-diagonal classical noise or combinations of quantum and classical
        noise. See the documentation for details.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_semiclassical(tspan::Vector{Float64}, rho0::State{DenseOperator},
                fquantum::Function, fclassical::Function;
                fstoch_quantum::Union{Function, Nothing}=nothing,
                fstoch_classical::Union{Function, Nothing}=nothing,
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                noise_processes::Int=0,
                noise_prototype_classical=nothing,
                nonlinear::Bool=true,
                kwargs...)

    tmp = copy(rho0.quantum)
    if isa(fstoch_quantum, Nothing) && isa(fstoch_classical, Nothing)
        throw(ArgumentError("No stochastic functions provided!"))
    end

    if noise_processes == 0
        n = 0
        if isa(fstoch_quantum, Function)
            fq_out = fstoch_quantum(0, rho0.quantum, rho0.classical)
            n += length(fq_out[1])
        end
    else
        n = noise_processes
    end

    if n > 0 && isa(fstoch_classical, Function)
        if isa(noise_prototype_classical, Nothing)
            throw(ArgumentError("noise_prototype_classical must be set for combinations of quantum and classical noise!"))
        end
    end

    dmaster_determ(t::Float64, rho::State{DenseOperator}, drho::State{DenseOperator}) =
            dmaster_h_dynamic(t, rho, fquantum, fclassical, rates, drho, tmp)

    dmaster_stoch(dx::DiffArray, t::Float64, rho::State{DenseOperator},
                    drho::State{DenseOperator}, n::Int) =
        dmaster_stoch_dynamic(dx, t, rho, fstoch_quantum, fstoch_classical, drho, n)

    integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch, rho0, fout, n;
                noise_prototype_classical=noise_prototype_classical,
                kwargs...)
end
master_semiclassical(tspan::Vector{Float64}, psi0::State{T}, args...; kwargs...) where T<:Ket =
        master_semiclassical(tspan, dm(psi0), args...; kwargs...)

# Derivative functions
function dschroedinger_stochastic(dx::Vector{ComplexF64}, t::Float64,
        state::State{T}, fstoch_quantum::Function, fstoch_classical::Nothing,
        dstate::State{T}, ::Int) where T<:Ket
    H = fstoch_quantum(t, state.quantum, state.classical)
    recast!(dx, dstate)
    QO_CHECKS[] && check_schroedinger(state.quantum, H[1])
    dschroedinger(state.quantum, H[1], dstate.quantum)
    recast!(dstate, dx)
end
function dschroedinger_stochastic(dx::Array{ComplexF64, 2},
        t::Float64, state::State{T}, fstoch_quantum::Function,
        fstoch_classical::Nothing, dstate::State{T}, n::Int) where T<:Ket
    H = fstoch_quantum(t, state.quantum, state.classical)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, dstate)
        QO_CHECKS[] && check_schroedinger(state.quantum, H[i])
        dschroedinger(state.quantum, H[i], dstate.quantum)
        recast!(dstate, dx_i)
    end
end
function dschroedinger_stochastic(dx::DiffArray, t::Float64,
            state::State{T}, fstoch_quantum::Nothing, fstoch_classical::Function,
            dstate::State{T}, ::Int) where T<:Ket
    dclassical = @view dx[length(state.quantum)+1:end, :]
    fstoch_classical(t, state.quantum, state.classical, dclassical)
end
function dschroedinger_stochastic(dx::Array{ComplexF64, 2}, t::Float64, state::State{T}, fstoch_quantum::Function,
            fstoch_classical::Function, dstate::State{T}, n::Int) where T<:Ket
    dschroedinger_stochastic(dx, t, state, fstoch_quantum, nothing, dstate, n)

    dx_i = @view dx[length(state.quantum)+1:end, n+1:end]
    fstoch_classical(t, state.quantum, state.classical, dx_i)
end

function dmaster_stoch_dynamic(dx::Vector{ComplexF64}, t::Float64,
            state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Nothing, dstate::State{DenseOperator}, ::Int)
    result = fstoch_quantum(t, state.quantum, state.classical)
    QO_CHECKS[] && @assert length(result) == 2
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(state.quantum, C, Cdagger)
    recast!(dx, dstate)
    operators.gemm!(1, C[1], state.quantum, 0, dstate.quantum)
    operators.gemm!(1, state.quantum, Cdagger[1], 1, dstate.quantum)
    dstate.quantum.data .-= tr(dstate.quantum)*state.quantum.data
    recast!(dstate, dx)
end
function dmaster_stoch_dynamic(dx::Array{ComplexF64, 2}, t::Float64,
            state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Nothing, dstate::State{DenseOperator}, n::Int)
    result = fstoch_quantum(t, state.quantum, state.classical)
    QO_CHECKS[] && @assert length(result) == 2
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(state.quantum, C, Cdagger)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, dstate)
        operators.gemm!(1, C[i], state.quantum, 0, dstate.quantum)
        operators.gemm!(1, state.quantum, Cdagger[i], 1, dstate.quantum)
        dstate.quantum.data .-= tr(dstate.quantum)*state.quantum.data
        recast!(dstate, dx_i)
    end
end
function dmaster_stoch_dynamic(dx::DiffArray, t::Float64,
            state::State{DenseOperator}, fstoch_quantum::Nothing,
            fstoch_classical::Function, dstate::State{DenseOperator}, ::Int)
    dclassical = @view dx[length(state.quantum)+1:end, :]
    fstoch_classical(t, state.quantum, state.classical, dclassical)
end
function dmaster_stoch_dynamic(dx::Array{ComplexF64, 2}, t::Float64,
            state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Function, dstate::State{DenseOperator}, n::Int)
    dmaster_stoch_dynamic(dx, t, state, fstoch_quantum, nothing, dstate, n)

    dx_i = @view dx[length(state.quantum)+1:end, n+1:end]
    fstoch_classical(t, state.quantum, state.classical, dx_i)
end

function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::State{DenseOperator}, fout::Union{Nothing, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = Vector{ComplexF64}(undef, length(rho0))
    recast!(rho0, x0)
    state = copy(rho0)
    dstate = copy(rho0)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function recast!(state::State, x::SubArray{ComplexF64, 1})
    N = length(state.quantum)
    copyto!(x, 1, state.quantum.data, 1, N)
    copyto!(x, N+1, state.classical, 1, length(state.classical))
    x
end
function recast!(x::SubArray{ComplexF64, 1}, state::State)
    N = length(state.quantum)
    copyto!(state.quantum.data, 1, x, 1, N)
    copyto!(state.classical, 1, x, N+1, length(state.classical))
end

end # module
