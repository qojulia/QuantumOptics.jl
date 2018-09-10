module timeevolution_schroedinger

export schroedinger, schroedinger_dynamic

import ..integrate, ..recast!, ..QO_CHECKS

using ...bases, ...states, ...operators


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
function schroedinger(tspan, psi0::T, H::AbstractOperator;
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where T<:StateVector
    tspan_ = convert(Vector{Float64}, tspan)
    check_schroedinger(psi0, H)
    dschroedinger_(t::Float64, psi::T, dpsi::T) = dschroedinger(psi, H, dpsi)
    x0 = psi0.data
    state = T(psi0.basis, psi0.data)
    dstate = T(psi0.basis, psi0.data)
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
                kwargs...) where T<:StateVector
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_(t::Float64, psi::T, dpsi::T) = dschroedinger_dynamic(t, psi, f, dpsi)
    x0 = psi0.data
    state = Ket(psi0.basis, psi0.data)
    dstate = Ket(psi0.basis, psi0.data)
    integrate(tspan_, dschroedinger_, x0, state, dstate, fout; kwargs...)
end


recast!(x::Vector{ComplexF64}, psi::StateVector) = (psi.data = x);
recast!(psi::StateVector, x::Vector{ComplexF64}) = nothing


function dschroedinger(psi::Ket, H::AbstractOperator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    return dpsi
end

function dschroedinger(psi::Bra, H::AbstractOperator, dpsi::Bra)
    operators.gemv!(complex(0,1.), psi, H, complex(0.), dpsi)
    return dpsi
end


function dschroedinger_dynamic(t::Float64, psi0::T, f::Function, dpsi::T) where T<:StateVector
    H = f(t, psi0)
    QO_CHECKS[] && check_schroedinger(psi0, H)
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

end
