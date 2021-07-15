using ...semiclassical
import ...semiclassical: State

"""
    stochastic.schroedinger_semiclassical(tspan, state0, fquantum, fclassical[; fout, ...])
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
function schroedinger_semiclassical(tspan, state0::S, fquantum,
                fclassical; fstoch_quantum=nothing,
                fstoch_classical=nothing,
                fout=nothing,
                noise_processes::Int=0,
                noise_prototype_classical=nothing,
                normalize_state::Bool=false,
                kwargs...) where {B<:Basis,T<:Ket{B},S<:State{B,T}}
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    dschroedinger_det(t, state, dstate) =
            semiclassical.dschroedinger_dynamic!(dstate, fquantum, fclassical, state, t)

    if isa(fstoch_quantum, Nothing) && isa(fstoch_classical, Nothing)
        throw(ArgumentError("No stochastic functions provided!"))
    end

    x0 = Vector{eltype(state0)}(undef, length(state0))
    recast!(x0,state0)
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
        function norm_func(u, t, integrator)
            u .= [normalize!(u[1:len_q]); u[len_q+1:end]]
        end
        ncb = DiffEqCallbacks.FunctionCallingCallback(norm_func;
                 func_everystep=true,
                 func_start=false)
    else
        ncb = nothing
    end

    dschroedinger_stoch(dx, t, state, dstate, n) =
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
function master_semiclassical(tspan, rho0::S,
                fquantum, fclassical;
                fstoch_quantum=nothing,
                fstoch_classical=nothing,
                rates=nothing,
                fout=nothing,
                noise_processes::Int=0,
                noise_prototype_classical=nothing,
                nonlinear::Bool=true,
                kwargs...) where {B<:Basis,T<:Operator{B,B},S<:State{B,T}}

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

    dmaster_determ(t, rho, drho) =
            semiclassical.dmaster_h_dynamic!(drho, fquantum, fclassical, rates, rho, tmp, t)

    dmaster_stoch(dx, t, rho, drho, n) =
        dmaster_stoch_dynamic(dx, t, rho, fstoch_quantum, fstoch_classical, drho, n)

    integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch, rho0, fout, n;
                noise_prototype_classical=noise_prototype_classical,
                kwargs...)
end
master_semiclassical(tspan, psi0::State{B,T}, args...; kwargs...) where {B<:Basis,T<:Ket{B}} =
        master_semiclassical(tspan, dm(psi0), args...; kwargs...)

# Derivative functions
function dschroedinger_stochastic(dx::AbstractVector, t,
        state, fstoch_quantum::Function, fstoch_classical::Nothing,
        dstate, n)
    H = fstoch_quantum(t, state.quantum, state.classical)
    recast!(dstate,dx)
    QO_CHECKS[] && check_schroedinger(state.quantum, H[1])
    dschroedinger!(dstate.quantum, H[1], state.quantum)
    recast!(dx,dstate)
end
function dschroedinger_stochastic(dx::AbstractMatrix, t,
        state, fstoch_quantum::Function, fstoch_classical::Nothing,
        dstate, n)
    H = fstoch_quantum(t, state.quantum, state.classical)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dstate,dx_i)
        QO_CHECKS[] && check_schroedinger(state.quantum, H[i])
        dschroedinger!(dstate.quantum, H[i], state.quantum)
        recast!(dx_i,dstate)
    end
end
function dschroedinger_stochastic(dx, t,
            state, fstoch_quantum::Nothing, fstoch_classical::Function,
            dstate, n)
    dclassical = @view dx[length(state.quantum)+1:end, :]
    fstoch_classical(t, state.quantum, state.classical, dclassical)
end
function dschroedinger_stochastic(dx, t, state, fstoch_quantum::Function,
            fstoch_classical::Function, dstate, n)
    dschroedinger_stochastic(dx, t, state, fstoch_quantum, nothing, dstate, n)

    dx_i = @view dx[length(state.quantum)+1:end, n+1:end]
    fstoch_classical(t, state.quantum, state.classical, dx_i)
end

function dmaster_stoch_dynamic(dx::AbstractVector, t,
            state, fstoch_quantum::Function,
            fstoch_classical::Nothing, dstate, n)
    result = fstoch_quantum(t, state.quantum, state.classical)
    QO_CHECKS[] && @assert length(result) == 2
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(state.quantum, C, Cdagger)
    recast!(dstate,dx)
    QuantumOpticsBase.mul!(dstate.quantum,C[1],state.quantum)
    QuantumOpticsBase.mul!(dstate.quantum,state.quantum,Cdagger[1],true,true)
    dstate.quantum.data .-= tr(dstate.quantum)*state.quantum.data
    recast!(dx,dstate)
end
function dmaster_stoch_dynamic(dx::AbstractMatrix, t,
            state, fstoch_quantum::Function,
            fstoch_classical::Nothing, dstate, n)
    result = fstoch_quantum(t, state.quantum, state.classical)
    QO_CHECKS[] && @assert length(result) == 2
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(state.quantum, C, Cdagger)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dstate,dx_i)
        QuantumOpticsBase.mul!(dstate.quantum,C[i],state.quantum)
        QuantumOpticsBase.mul!(dstate.quantum,state.quantum,Cdagger[i],true,true)
        dstate.quantum.data .-= tr(dstate.quantum)*state.quantum.data
        recast!(dx_i,dstate)
    end
end
function dmaster_stoch_dynamic(dx, t,
            state, fstoch_quantum::Nothing,
            fstoch_classical::Function, dstate, n)
    dclassical = @view dx[length(state.quantum)+1:end, :]
    fstoch_classical(t, state.quantum, state.classical, dclassical)
end
function dmaster_stoch_dynamic(dx, t,
            state, fstoch_quantum::Function,
            fstoch_classical::Function, dstate, n)
    dmaster_stoch_dynamic(dx, t, state, fstoch_quantum, nothing, dstate, n)

    dx_i = @view dx[length(state.quantum)+1:end, n+1:end]
    fstoch_classical(t, state.quantum, state.classical, dx_i)
end

function recast!(x::SubArray,state::State)
    N = length(state.quantum)
    copyto!(x, 1, state.quantum.data, 1, N)
    copyto!(x, N+1, state.classical, 1, length(state.classical))
    x
end
function recast!(state::State,x::SubArray)
    N = length(state.quantum)
    copyto!(state.quantum.data, 1, x, 1, N)
    copyto!(state.classical, 1, x, N+1, length(state.classical))
end
