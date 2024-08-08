module semiclassical

using QuantumOpticsBase
import Base: ==, isapprox, +, -, *, /
import ..timeevolution: integrate, recast!, jump, integrate_mcwf, jump_callback,
    JumpRNGState, threshold, roll!, as_vector, QO_CHECKS
import LinearAlgebra: normalize, normalize!
import RecursiveArrayTools

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
State{B}(q::T, c::C) where {B,T<:QuantumState{B},C} = State(q,c)

# Standard interfaces
Base.zero(x::State) = State(zero(x.quantum), zero(x.classical))
Base.length(x::State) = length(x.quantum) + length(x.classical)
Base.axes(x::State) = (Base.OneTo(length(x)),)
Base.size(x::State) = size(x.quantum)
Base.ndims(x::State) = ndims(x.quantum)
Base.ndims(x::Type{<:State{B,T,C}}) where {B,T<:QuantumState{B},C} = ndims(T)
Base.copy(x::State) = State(copy(x.quantum), copy(x.classical))
Base.copyto!(x::State, y::State) = (copyto!(x.quantum, y.quantum); copyto!(x.classical, y.classical); x)
Base.fill!(x::State, a) = (fill!(x.quantum, a), fill!(x.classical, a))
Base.eltype(x::State) = promote_type(eltype(x.quantum),eltype(x.classical))
Base.eltype(x::Type{<:State{B,T,C}}) where {B,T<:QuantumState{B},C} = promote_type(eltype(T), eltype(C))
Base.similar(x::State, ::Type{T} = eltype(x)) where {T} = State(similar(x.quantum, T), similar(x.classical, T))
Base.getindex(x::State, idx) = idx <= length(x.quantum) ? getindex(x.quantum, idx) : getindex(x.classical, idx-length(x.quantum))
Base.setindex!(x::State, v, idx) = idx <= length(x.quantum) ? setindex(x.quantum, v, idx) : setindex(x.classical, v, idx-length(x.quantum))

normalize!(x::State) = (normalize!(x.quantum); x)
normalize(x::State) = State(normalize(x.quantum),copy(x.classical))
LinearAlgebra.norm(x::State) = LinearAlgebra.norm(x.quantum)
LinearAlgebra.norm(x::State, p::Int64) = LinearAlgebra.norm(x.quantum, p)

==(x::State{B}, y::State{B}) where {B} = (x.classical==y.classical) && (x.quantum==y.quantum)
==(x::State, y::State) = false

isapprox(x::State{B}, y::State{B}; kwargs...) where {B} = isapprox(x.quantum,y.quantum) && isapprox(x.classical,y.classical)
isapprox(x::State, y::State; kwargs...) = false

QuantumOpticsBase.expect(op, state::State) = expect(op, state.quantum)
QuantumOpticsBase.variance(op, state::State) = variance(op, state.quantum)
QuantumOpticsBase.ptrace(state::State, indices) = State(ptrace(state.quantum, indices), state.classical)
QuantumOpticsBase.dm(x::State) = State(dm(x.quantum), x.classical)

Base.broadcastable(x::State) = x

# Custom broadcasting style
struct StateStyle{B} <: Broadcast.BroadcastStyle end

# Style precedence rules
Broadcast.BroadcastStyle(::Type{<:State{B}}) where {B} = StateStyle{B}()
Broadcast.BroadcastStyle(::StateStyle{B1}, ::StateStyle{B2}) where {B1,B2} = throw(IncompatibleBases())
Broadcast.BroadcastStyle(::StateStyle{B}, ::Broadcast.DefaultArrayStyle{0}) where {B} = StateStyle{B}()
Broadcast.BroadcastStyle(::Broadcast.DefaultArrayStyle{0}, ::StateStyle{B}) where {B} = StateStyle{B}()

# Out-of-place broadcasting
@inline function Base.copy(bc::Broadcast.Broadcasted{<:StateStyle{B},Axes,F,Args}) where {B,Axes,F,Args<:Tuple}
    bcf = Broadcast.flatten(bc)
    # extract quantum object from broadcast container
    qobj = find_quantum(bcf)
    data_q = zeros(eltype(qobj), size(qobj)...)
    Nq = length(qobj)
    # allocate quantum data from broadcast container
    @inbounds @simd for I in 1:Nq
        data_q[I] = bcf[I]
    end
    # extract classical object from broadcast container
    cobj = find_classical(bcf)
    data_c = zeros(eltype(cobj), length(cobj))
    Nc = length(cobj)
    # allocate classical data from broadcast container
    @inbounds @simd for I in 1:Nc
        data_c[I] = bcf[I+Nq]
    end
    type = eval(nameof(typeof(qobj)))
    return State{B}(type(basis(qobj), data_q), data_c)
end

for f ∈ [:find_quantum, :find_classical]
    @eval ($f)(bc::Broadcast.Broadcasted) = ($f)(bc.args)
    @eval ($f)(args::Tuple) = ($f)(($f)(args[1]), Base.tail(args))
    @eval ($f)(x) = x
    @eval ($f)(::Any, rest) = ($f)(rest)
end
find_quantum(x::State, rest) = x.quantum
find_classical(x::State, rest) = x.classical

# In-place broadcasting
@inline function Base.copyto!(dest::State{B}, bc::Broadcast.Broadcasted{<:StateStyle{B},Axes,F,Args}) where {B,Axes,F,Args<:Tuple}
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    # write broadcasted quantum data to dest
    qobj = dest.quantum
    @inbounds @simd for I in 1:length(qobj)
        qobj.data[I] = bc′[I]
    end
    # write broadcasted classical data to dest
    cobj = dest.classical
    @inbounds @simd for I in 1:length(cobj)
        cobj[I] = bc′[I+length(qobj)]
    end
    return dest
end
@inline Base.copyto!(dest::State{B1}, bc::Broadcast.Broadcasted{<:StateStyle{B2},Axes,F,Args}) where {B1,B2,Axes,F,Args<:Tuple} =
    throw(IncompatibleBases())

Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(x::State, i) = Base.getindex(x, i)
RecursiveArrayTools.recursive_unitless_bottom_eltype(x::State) = eltype(x)
RecursiveArrayTools.recursivecopy!(dest::State, src::State) = copyto!(dest, src)
RecursiveArrayTools.recursivecopy(x::State) = copy(x)
RecursiveArrayTools.recursivefill!(x::State, a) = fill!(x, a)

"""
    semiclassical.schroedinger_dynamic(tspan, state0, fquantum, fclassical[; fout, ...])

Integrate time-dependent Schrödinger equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, psi, u) -> H` returning the time and or state
        dependent Hamiltonian.
* `fclassical!`: Function `f!(du, u, psi, t)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, state0::State, fquantum, fclassical!;
                fout=nothing,
                kwargs...)
    dschroedinger_ = let fquantum = fquantum, fclassical! = fclassical!
        dschroedinger_(t, state, dstate) = dschroedinger_dynamic!(dstate, fquantum, fclassical!, state, t)
    end
    x0 = Vector{eltype(state0)}(undef, length(state0))
    recast!(x0,state0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end

"""
    semiclassical.master_dynamic(tspan, state0, fquantum, fclassical!; <keyword arguments>)

Integrate time-dependent master equation coupled to a classical system.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial semi-classical state [`semiclassical.State`](@ref).
* `fquantum`: Function `f(t, rho, u) -> (H, J, Jdagger)` returning the time
        and/or state dependent Hamiltonian and Jump operators.
* `fclassical!`: Function `f!(du, u, rho, t)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the complex vector `du`.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is not
        permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan, state0::State{B,T}, fquantum, fclassical!;
                rates=nothing,
                fout=nothing,
                tmp=copy(state0.quantum),
                kwargs...) where {B,T<:Operator}
    dmaster_ = let fquantum = fquantum, fclassical! = fclassical!
        dmaster_(t, state, dstate) = dmaster_h_dynamic!(dstate, fquantum, fclassical!, rates, state, tmp, t)
    end
    x0 = Vector{eltype(state0)}(undef, length(state0))
    recast!(x0,state0)
    state = copy(state0)
    dstate = copy(state0)
    integrate(tspan, dmaster_, x0, state, dstate, fout; kwargs...)
end

function master_dynamic(tspan, state0::State{B,T}, fquantum, fclassical!; kwargs...) where {B,T<:Ket{B}}
    master_dynamic(tspan, dm(state0), fquantum, fclassical!; kwargs...)
end

"""
    semiclassical.mcwf_dynamic(tspan, psi0, fquantum, fclassical!, fjump_classical!; <keyword arguments>)

Calculate MCWF trajectories coupled to a classical system. **NOTE**: The quantum
state with which `fquantum` and `fclassical!` are called is NOT NORMALIZED. Make
sure to take this into account when computing expectation values!

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `psi0`: Initial semi-classical state [`semiclassical.State`](@ref) featuring
        a `Ket`(@ref).
* `fquantum`: Function `f(t, psi, u) -> (H, J, Jdagger)` returning the time
        and/or state dependent Hamiltonian and Jump operators.
* `fclassical!`: Function `f!(du, u, psi, t)` calculating the possibly time and
        state dependent derivative of the classical equations and storing it
        in the complex vector `du`.
* `fjump_classical!`: Function `f(t, psi, u, i)` performing a classical jump when a
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
function mcwf_dynamic(tspan, psi0::State{B,T}, fquantum, fclassical!, fjump_classical!;
                seed=rand(UInt),
                rates=nothing,
                fout=nothing,
                kwargs...) where {B,T<:Ket}
    tmp=copy(psi0.quantum)
    dmcwf_ = let fquantum = fquantum, fclassical! = fclassical!
        dmcwf_(t, psi, dpsi) = dmcwf_h_dynamic!(dpsi, fquantum, fclassical!, rates, psi, tmp, t)
    end
    J = fquantum(first(tspan), psi0.quantum, psi0.classical)[2]
    probs = zeros(real(eltype(psi0)), length(J))
    j_ = let fquantum = fquantum, fclassical! = fclassical!, fjump_classical! = fjump_classical!, probs = probs
        j_(rng, t, psi, psi_new) = jump_dynamic(rng, t, psi, fquantum, fclassical!, fjump_classical!, psi_new, probs, rates)
    end
    x0 = Vector{eltype(psi0)}(undef, length(psi0))
    recast!(x0,psi0)
    psi = copy(psi0)
    dpsi = copy(psi0)
    integrate_mcwf(dmcwf_, j_, tspan, psi, seed, fout; kwargs...)
end

function recast!(x::C,state::State{B,T,C}) where {B,T,C}
    N = length(state.quantum)
    copyto!(x, 1, state.quantum.data, 1, N)
    copyto!(x, N+1, state.classical, 1, length(state.classical))
    x
end

function recast!(state::State{B,T,C},x::C) where {B,T,C}
    N = length(state.quantum)
    copyto!(state.quantum.data, 1, x, 1, N)
    copyto!(state.classical, 1, x, N+1, length(state.classical))
end

"""
    dschroedinger_dynamic!(dstate, fquantum, fclassical!, state)

Update the semiclassical state `dstate` according to a time-dependent,
semiclassical Schrödinger equation.

See also: [`semiclassical.schroedinger_dynamic`](@ref)
"""
function dschroedinger_dynamic!(dstate, fquantum::F, fclassical!::G, state, t) where {F,G}
    fquantum_(t, psi) = fquantum(t, state.quantum, state.classical)
    timeevolution.dschroedinger_dynamic!(dstate.quantum, fquantum_, state.quantum, t)
    fclassical!(dstate.classical, state.classical, state.quantum, t)
    return dstate
end

"""
    dmaster_h_dynamic!(dstate, fquantum, fclassical!, rates, state, tmp, t)

Update the semiclassical state `dstate` according to a time-dependent,
semiclassical master eqaution.

See also: [`semiclassical.master_dynamic`](@ref)
"""
function dmaster_h_dynamic!(dstate, fquantum::F, fclassical!::G, rates, state, tmp, t) where {F,G}
    fquantum_(t, rho) = fquantum(t, state.quantum, state.classical)
    timeevolution.dmaster_h_dynamic!(dstate.quantum, fquantum_, rates, state.quantum, tmp, t)
    fclassical!(dstate.classical, state.classical, state.quantum, t)
    return dstate
end

"""
    dmcwf_h_dynamic!(dpsi, fquantum, fclassical!, rates, psi, tmp, t)

Update the semiclassical state `dpsi` according to a time-dependent, semiclassical
and non-Hermitian Schrödinger equation (MCWF).

See also: [`semiclassical.mcwf_dynamic`](@ref)
"""
function dmcwf_h_dynamic!(dpsi, fquantum::F, fclassical!::G, rates, psi, tmp, t) where {F,G}
    fquantum_(t, rho) = fquantum(t, psi.quantum, psi.classical)
    timeevolution.dmcwf_h_dynamic!(dpsi.quantum, fquantum_, rates, psi.quantum, tmp, t)
    fclassical!(dpsi.classical, psi.classical, psi.quantum, t)
    return dpsi
end

function jump_dynamic(rng, t, psi, fquantum::F, fclassical!::G, fjump_classical!::H, psi_new, probs_tmp, rates) where {F,G,H}
    result = fquantum(t, psi.quantum, psi.classical)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    J = result[2]
    if length(result) == 3
        rates_ = rates
    else
        rates_ = result[4]
    end
    i = jump(rng, t, psi.quantum, J, psi_new.quantum, probs_tmp, rates_)
    fjump_classical!(psi.classical, psi_new.quantum, i, t)
    psi_new.classical .= psi.classical
    return i
end

function jump_callback(jumpfun::F, seed, scb, save_before!::G,
                        save_after!::H, save_t_index::I, psi0::State, rng_state::JumpRNGState) where {F,G,H,I}
    tmp = copy(psi0)
    psi_tmp = copy(psi0)

    n = length(psi0.quantum)
    djumpnorm(x::Vector, t, integrator) = norm(x[1:n])^2 - (1-threshold(rng_state))

    function dojump(integrator)
        x = integrator.u
        t = integrator.t

        affect! = scb.affect!
        save_before!(affect!,integrator)
        recast!(psi_tmp,x)
        i = jumpfun(rng_state.rng, t, psi_tmp, tmp)
        recast!(x,tmp)
        save_after!(affect!,integrator)
        save_t_index(t,i)

        roll!(rng_state)
        return nothing
    end

    return OrdinaryDiffEq.ContinuousCallback(djumpnorm,dojump,
                     save_positions = (false,false))
end
as_vector(psi::State) = vcat(psi.quantum.data[:], psi.classical)


end # module
