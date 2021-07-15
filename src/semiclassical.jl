module semiclassical

using QuantumOpticsBase
import Base: ==
import ..timeevolution: integrate, recast!, jump, integrate_mcwf, jump_callback, as_vector, QO_CHECKS
import LinearAlgebra: normalize, normalize!

using Random, LinearAlgebra
import OrdinaryDiffEq

# TODO: Remove imports
import DiffEqCallbacks, RecursiveArrayTools.copyat_or_push!
Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

using ..timeevolution


const QuantumState{B} = Union{Ket{B}, Operator{B,B}}

"""
Semi-classical state.

It consists of a quantum part, which is either a `Ket` or a `DenseOperator` and
a classical part that is specified as a complex vector of arbitrary length.
"""
mutable struct State{B,T,C}
    quantum::T
    classical::C
    function State(quantum::T, classical::C) where {B,T<:QuantumState{B},C}
        new{B,T,C}(quantum, classical)
    end
end

Base.length(state::State) = length(state.quantum) + length(state.classical)
Base.copy(state::State) = State(copy(state.quantum), copy(state.classical))
Base.eltype(state::State) = promote_type(eltype(state.quantum),eltype(state.classical))
normalize!(state::State) where {B,T} = (normalize!(state.quantum); state)
normalize(state::State) = State(normalize(state.quantum),copy(state.classical))

function ==(a::State, b::State)
    QuantumOpticsBase.samebases(a.quantum, b.quantum) &&
    length(a.classical)==length(b.classical) &&
    (a.classical==b.classical) &&
    (a.quantum==b.quantum)
end

QuantumOpticsBase.expect(op, state::State) = expect(op, state.quantum)
QuantumOpticsBase.variance(op, state::State) = variance(op, state.quantum)
QuantumOpticsBase.ptrace(state::State, indices) = State(ptrace(state.quantum, indices), state.classical)

QuantumOpticsBase.dm(x::State) = State(dm(x.quantum), x.classical)


"""
    semiclassical.schroedinger_dynamic(tspan, state0, fquantum, fclassical[; fout, ...])

Integrate time-dependent Schrödinger equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, psi, u) -> H` returning the time and or state
        dependent Hamiltonian.
* `fclassical`: Function `f(t, psi, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, state0::State, fquantum, fclassical;
                fout=nothing,
                kwargs...)
    dschroedinger_(t, state, dstate) = dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)
    x0 = Vector{eltype(state0)}(undef, length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end

"""
    semiclassical.master_dynamic(tspan, state0, fquantum, fclassical; <keyword arguments>)

Integrate time-dependent master equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, rho, u) -> (H, J, Jdagger)` returning the time
        and/or state dependent Hamiltonian and Jump operators.
* `fclassical`: Function `f(t, rho, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the complex vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is not
        permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan, state0::State{B,T}, fquantum, fclassical;
                rates=nothing,
                fout=nothing,
                tmp=copy(state0.quantum),
                kwargs...) where {B,T<:Operator}
    dmaster_(t, state, dstate) = dmaster_h_dynamic(t, state, fquantum, fclassical, rates, dstate, tmp)
    x0 = Vector{eltype(state0)}(undef, length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan, dmaster_, x0, state, dstate, fout; kwargs...)
end

function master_dynamic(tspan, state0::State{B,T}, fquantum, fclassical; kwargs...) where {B,T<:Ket{B}}
    master_dynamic(tspan, dm(state0), fquantum, fclassical; kwargs...)
end

"""
    semiclassical.mcwf_dynamic(tspan, psi0, fquantum, fclassical, fjump_classical; <keyword arguments>)

Calculate MCWF trajectories coupled to a classical system. **NOTE**: The quantum
state with which `fquantum` and `fclassical` are called is NOT NORMALIZED. Make
sure to take this into account when computing expectation values!

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref) featuring
        a `Ket`(@ref).
* `fquantum`: Function `f(t, psi, u) -> (H, J, Jdagger)` returning the time
        and/or state dependent Hamiltonian and Jump operators.
* `fclassical`: Function `f(t, psi, u, du)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the complex vector `du`.
* `fjump_classical`: Function `f(t, psi, u, i)` performing a classical jump when a
        quantum jump of the i-th jump operator occurs.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is not
        permanent!
* `display_beforeevent`: Choose whether or not an additional point should be saved
        before a jump occurs. Default is false.
* `display_afterevent`: Choose whether or not an additional point should be saved
        after a jump occurs. Default is false.
* `display_jumps=false`: If set to true, an additional list of times and indices
        is returned. These correspond to the times at which a jump occured and
        the index of the jump operators with which the jump occured, respectively.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf_dynamic(tspan, psi0::State{B,T}, fquantum, fclassical, fjump_classical;
                seed=rand(UInt),
                rates=nothing,
                fout=nothing,
                kwargs...) where {B,T<:Ket}
    tmp=copy(psi0.quantum)
    dmcwf_(t, psi, dpsi) = dmcwf_h_dynamic(t, psi, fquantum, fclassical, rates, dpsi, tmp)
    j_(rng, t, psi, psi_new) = jump_dynamic(rng, t, psi, fquantum, fclassical, fjump_classical, psi_new, rates)
    x0 = Vector{eltype(psi0)}(undef, length(psi0))
    recast!(psi0, x0)
    psi = copy(psi0)
    dpsi = copy(psi0)
    integrate_mcwf(dmcwf_, j_, tspan, psi, seed, fout; kwargs...)
end

function recast!(state::State{B,T,C}, x::C) where {B,T,C}
    N = length(state.quantum)
    copyto!(x, 1, state.quantum.data, 1, N)
    copyto!(x, N+1, state.classical, 1, length(state.classical))
    x
end

function recast!(x::C, state::State{B,T,C}) where {B,T,C}
    N = length(state.quantum)
    copyto!(state.quantum.data, 1, x, 1, N)
    copyto!(state.classical, 1, x, N+1, length(state.classical))
end

function dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)
    fquantum_(t, psi) = fquantum(t, state.quantum, state.classical)
    timeevolution.dschroedinger_dynamic!(dstate.quantum, fquantum_, state.quantum, t)
    fclassical(t, state.quantum, state.classical, dstate.classical)
end

function dmaster_h_dynamic(t, state, fquantum, fclassical, rates, dstate, tmp)
    fquantum_(t, rho) = fquantum(t, state.quantum, state.classical)
    timeevolution.dmaster_h_dynamic!(dstate.quantum, fquantum_, rates, state.quantum, tmp, t)
    fclassical(t, state.quantum, state.classical, dstate.classical)
end

function dmcwf_h_dynamic(t, psi, fquantum, fclassical, rates, dpsi, tmp)
    fquantum_(t, rho) = fquantum(t, psi.quantum, psi.classical)
    timeevolution.dmcwf_h_dynamic!(dpsi.quantum, fquantum_, rates, psi.quantum, tmp, t)
    fclassical(t, psi.quantum, psi.classical, dpsi.classical)
end

function jump_dynamic(rng, t, psi, fquantum, fclassical, fjump_classical, psi_new, rates)
    result = fquantum(t, psi.quantum, psi.classical)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    J = result[2]
    if length(result) == 3
        rates_ = rates
    else
        rates_ = result[4]
    end
    i = jump(rng, t, psi.quantum, J, psi_new.quantum, rates_)
    fjump_classical(t, psi_new.quantum, psi.classical, i)
    psi_new.classical .= psi.classical
    return i
end

function jump_callback(jumpfun, seed, scb, save_before!,
                        save_after!, save_t_index, psi0::State)
    tmp = copy(psi0)
    psi_tmp = copy(psi0)
    rng = MersenneTwister(convert(UInt, seed))
    jumpnorm = Ref(rand(rng))
    n = length(psi0.quantum)
    djumpnorm(x::Vector, t, integrator) = norm(x[1:n])^2 - (1-jumpnorm[])

    function dojump(integrator)
        x = integrator.u
        t = integrator.t

        affect! = scb.affect!
        save_before!(affect!,integrator)
        recast!(x, psi_tmp)
        i = jumpfun(rng, t, psi_tmp, tmp)
        recast!(tmp, x)
        save_after!(affect!,integrator)
        save_t_index(t,i)

        jumpnorm[] = rand(rng)
        return nothing
    end

    return OrdinaryDiffEq.ContinuousCallback(djumpnorm,dojump,
                     save_positions = (false,false))
end
as_vector(psi::State) = vcat(psi.quantum.data[:], psi.classical)


end # module
