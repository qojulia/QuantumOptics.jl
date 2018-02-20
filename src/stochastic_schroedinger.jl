module stochastic_schroedinger

export schroedinger, schroedinger_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...timeevolution
import ...timeevolution: integrate_stoch, recast!
import ...timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic, check_schroedinger

"""
    stochastic.schroedinger(tspan, state0, H, Hs[; fout, ...])

Integrate stochastic Schrödinger equation.

# Arguments
* `tspan`: Vector specifying the points of time for which the output should
        be displayed.
* `psi0`: Initial state as Ket.
* `H`: Deterministic part of the Hamiltonian.
* `Hs`: Stochastic part(s) of the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger(tspan, psi0::Ket, H::Operator, Hs::Vector;
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)

    n = length(Hs)
    dstate = copy(psi0)
    x0 = psi0.data
    state = copy(psi0)

    check_schroedinger(psi0, H)
    dschroedinger_determ(t::Float64, psi::Ket, dpsi::Ket) = dschroedinger(psi, H, dpsi)
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket, index::Int) = dschroedinger_stochastic(psi, Hs, dpsi, index)

    integrate_stoch(tspan_, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
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
* `fstoch`: Function or vector of functions `f(t, psi, u, du)` returning the stochastic part
    of the Hamiltonian.
* `fout=nothing`: If given, this function `fout(t, state)` is called every time
        an output should be displayed. ATTENTION: The given state is neither
        normalized nor permanent!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_dynamic(tspan, psi0::Ket, fdeterm::Function, fstoch::Function;
                fout::Union{Function,Void}=nothing, noise_processes::Int=0,
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
    dschroedinger_stoch(t::Float64, psi::Ket, dpsi::Ket, index::Int) =
        dschroedinger_stochastic(t, psi, fstoch, dpsi, index)

    integrate_stoch(tspan, dschroedinger_determ, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
end

function dschroedinger_stochastic(psi::Ket, Hs::Vector{T}, dpsi::Ket, index::Int) where T <: Operator
    check_schroedinger(psi, Hs[index])
    dschroedinger(psi, Hs[index], dpsi)
end

function dschroedinger_stochastic(t::Float64, psi::Ket, f::Function, dpsi::Ket)
    ops = f(t, psi)[1]
    check_schroedinger(psi, ops)
    dschroedinger(psi, ops, dpsi)
end
function dschroedinger_stochastic(t::Float64, psi::Ket, f::Function, dpsi::Ket, index::Int)
    ops = f(t, psi)
    check_schroedinger(psi, ops[index])
    dschroedinger(psi, ops[index], dpsi)
end

recast!(psi::StateVector, x::SubArray{Complex128, 1}) = (x .= psi.data)
recast!(x::SubArray{Complex128, 1}, psi::StateVector) = (psi.data = x)

end # module