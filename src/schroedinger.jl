module timeevolution_schroedinger

export schroedinger, schroedinger_dynamic

using ...bases
using ...states
using ...operators
using ...ode_dopri

"""
    integrate_schroedinger(dschroedinger, tspan, psi0; kwargs...)

Integrate Schroedinger equation with given derivative function.

# Arguments
* `dmaster::Function`: A function `f(t, psi, dpsi)` that calculates the
        time-derivative of `rho` at time `t` and stores the result in `dpsi`.
* `tspan::Vector`: Vector specifying the points of time for which output
        should be displayed.
* `psi0::DenseOperator`: Initial state.
* `fout::Function = nothing`: If given, this function `fout(t, rho)` is called
        every time an output should be displayed. ATTENTION: The state `psi` is
        not permanent! It is still in use by the ode solver and therefore must
        not be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function integrate_schroedinger{T<:StateVector}(dschroedinger::Function, tspan, psi0::T; fout=nothing, kwargs...)
    as_statevector(x::Vector{Complex128}) = T(psi0.basis, x)
    as_vector(psi::T) = psi.data
    if fout==nothing
        tout = Float64[]
        xout = T[]
        function fout_(t, psi::T)
            push!(tout, t)
            push!(xout, copy(psi))
        end
        f = fout_
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_statevector(x))
    dschroedinger_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dschroedinger(t, as_statevector(x), as_statevector(dx))
    ode(dschroedinger_, float(tspan), as_vector(psi0), f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function dschroedinger(psi::Ket, H::Operator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    return dpsi
end

function dschroedinger(psi::Bra, H::Operator, dpsi::Bra)
    operators.gemv!(complex(0,1.), psi, H, complex(0.), dpsi)
    return dpsi
end

function dschroedinger_dynamic{T<:StateVector}(t::Float64, psi0::T, f::Function, dpsi::T)
    H = f(t, psi0)
    _check_input(psi0, H)
    dschroedinger(psi0, H, dpsi)
end

function _check_input(psi::Ket, H::Operator)
    check_multiplicable(H, psi)
    check_samebases(H)
end

function _check_input(psi::Bra, H::Operator)
    check_multiplicable(psi, H)
    check_samebases(H)
end

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
function schroedinger{T<:StateVector}(tspan, psi0::T, H::Operator;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    _check_input(psi0, H)
    dschroedinger_(t, psi::T, dpsi::T) = dschroedinger(psi, H, dpsi)
    integrate_schroedinger(dschroedinger_, tspan, psi0; fout=fout, kwargs...)
end


"""
    timeevolution.schroedinger(tspan, psi0, f; fout)

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
function schroedinger_dynamic{T<:StateVector}(tspan, psi0::T, f::Function;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    dschroedinger_(t, psi::T, dpsi::T) = dschroedinger_dynamic(t, psi, f, dpsi)
    integrate_schroedinger(dschroedinger_, tspan, psi0; fout=fout, kwargs...)
end

end