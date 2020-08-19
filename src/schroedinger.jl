"""
    timeevolution.schroedinger(tspan, psi0, H; fout)

Integrate Schroedinger equation.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket).
* `H`: Arbitrary operator specifying the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.
"""
function schroedinger(tspan, psi0::T, H::AbstractOperator{B,B};
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Union{AbstractOperator{B,B},StateVector{B}}}
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    dschroedinger_(t, psi::T, dpsi::T) = dschroedinger(psi, H, dpsi)
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan_, dschroedinger_, x0, state, dstate, fout; kwargs...)
end


"""
    timeevolution.schroedinger_dynamic(tspan, psi0, f; fout)

Integrate time-dependent Schroedinger equation.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `psi0`: Initial state vector (can be a bra or a ket).
* `f`: Function `f(t, psi) -> H` returning the time and or state dependent Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, psi)` is called every time
        an output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver and
        therefore must not be changed.
"""
function schroedinger_dynamic(tspan, psi0::T, f::Function;
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where T<:Union{StateVector,AbstractOperator}
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    dschroedinger_(t, psi::T, dpsi::T) = dschroedinger_dynamic(t, psi, f, dpsi)
    x0 = psi0.data
    state = copy(psi0)
    dstate = copy(psi0)
    integrate(tspan_, dschroedinger_, x0, state, dstate, fout; kwargs...)
end


recast!(x::D, psi::StateVector{B,D}) where {B<:Basis, D} = (psi.data = x);
recast!(psi::StateVector{B,D}, x::D) where {B<:Basis, D} = nothing


function dschroedinger(psi::Union{Ket{B},AbstractOperator{B,B}}, H::AbstractOperator{B,B}, dpsi::Union{Ket{B},AbstractOperator{B,B}}) where B<:Basis
    QuantumOpticsBase.mul!(dpsi,H,psi,eltype(psi)(-im),zero(eltype(psi)))
    return dpsi
end

function dschroedinger(psi::Bra{B}, H::AbstractOperator{B,B}, dpsi::Bra{B}) where B<:Basis
    QuantumOpticsBase.mul!(dpsi,psi,H,eltype(psi)(im),zero(eltype(psi)))
    return dpsi
end


function dschroedinger_dynamic(t, psi0::T, f::Function, dpsi::T) where T<:Union{StateVector,AbstractOperator}
    H = f(t, psi0)
    dschroedinger(psi0, H, dpsi)
end


function check_schroedinger(psi::Ket, H::AbstractOperator)
    check_multiplicable(H, psi)
    check_samebases(H)
end

function check_schroedinger(psi::Bra, H::AbstractOperator)
    check_multiplicable(psi, H)
    check_samebases(H)
end
