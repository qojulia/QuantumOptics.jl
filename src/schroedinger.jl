"""
    timeevolution.schroedinger(tspan, psi0, H; fout)

Integrate Schroedinger equation to evolve states or compute propagators.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket) or initial propagator.
* `H`: Arbitrary operator specifying the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.
"""
function schroedinger(tspan, psi0::T, H::AbstractOperator{B,B};
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B,T<:Union{AbstractOperator{B,B},StateVector{B}}}
    dschroedinger_(t, psi, dpsi) = dschroedinger!(dpsi, H, psi)
    tspan, psi0 = _promote_time_and_state(tspan, psi0, H) # promote
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end


"""
    timeevolution.schroedinger_dynamic(tspan, psi0, f; fout)

Integrate time-dependent Schroedinger equation to evolve states or compute propagators.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket) or initial propagator.
* `f`: Function `f(t, psi) -> H` returning the time and or state dependent Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.
"""
function schroedinger_dynamic(tspan, psi0, f;
                fout::Union{Function,Nothing}=nothing,
                kwargs...)
    dschroedinger_(t, psi, dpsi) = dschroedinger_dynamic!(dpsi, f, psi, t)
    tspan, psi0 = _promote_time_and_state(tspan, psi0, f) # promote
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan, dschroedinger_, x0, state, dstate, fout; kwargs...)
end

"""
    recast!(x,y)

Write the data stored in `y` into `x`, where either `x` or `y` is a quantum
object such as a [`Ket`](@ref) or an [`Operator`](@ref), and the other one is
a vector or a matrix with a matching size.
"""
recast!(psi::StateVector{B,D},x::D) where {B, D} = (psi.data = x);
recast!(x::D,psi::StateVector{B,D}) where {B, D} = nothing
function recast!(proj::Operator{B1,B2,T},x::T) where {B1,B2,T}
    proj.data = x
end
recast!(x::T,proj::Operator{B1,B2,T}) where {B1,B2,T} = nothing

"""
    dschroedinger!(dpsi, H, psi)

Update the increment `dpsi` in-place according to a Schrödinger equation
as `-im*H*psi`.

See also: [`dschroedinger_dynamic!`](@ref)
"""
function dschroedinger!(dpsi, H, psi)
    QuantumOpticsBase.mul!(dpsi,H,psi,eltype(psi)(-im),zero(eltype(psi)))
    return dpsi
end

function dschroedinger!(dpsi, H, psi::Bra)
    QuantumOpticsBase.mul!(dpsi,psi,H,eltype(psi)(im),zero(eltype(psi)))
    return dpsi
end

"""
    dschroedinger_dynamic!(dpsi, f, psi, t)

Compute the Hamiltonian as `H=f(t, psi)` and update `dpsi` according to a
Schrödinger equation as `-im*H*psi`.

See also: [`dschroedinger!`](@ref)
"""
function dschroedinger_dynamic!(dpsi, f, psi, t)
    H = f(t, psi)
    dschroedinger!(dpsi, H, psi)
end


function check_schroedinger(psi, H)
    check_multiplicable(H, psi)
    check_samebases(H)
end

function check_schroedinger(psi::Bra, H)
    check_multiplicable(psi, H)
    check_samebases(H)
end


_promote_time_and_state(tspan, psi0, f) = _promote_time_and_state(tspan, psi0, f(first(tspan), psi0))
function _promote_time_and_state(tspan, psi0, H::AbstractOperator)
    # general case is Ts<:Complex, Tt<:Real
    Ts = eltype(H)
    Tt = real(Ts)
    (isconcretetype(Ts) && isconcretetype(Tt)) || @warn "For using `ForwardDiff` on `schroedinger` the element type of `real(H(t,psi)*psi)` must be concrete !!\nGot elements of type $Tt \nTry promoting the Hamiltonian elements based on the parameters you are differentiating by."
    tspan = Tt.(tspan)
    psi0 = _promote_state(Ts, psi0)
    return tspan, psi0
end
_promote_state(Ts, psi0::Operator) = Operator(psi0.basis_l, psi0.basis_r, Ts.(psi0.data))
_promote_state(Ts, psi0::Ket) = Ket(psi0.basis, Ts.(psi0.data))