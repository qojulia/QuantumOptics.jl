module stochastic_semiclassical

export schroedinger_semiclassical, master_semiclassical

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...semiclassical
import ...semiclassical: recast!, State, dmaster_h_dynamic
using ...timeevolution
import ...timeevolution: integrate_stoch
import ...timeevolution.timeevolution_schroedinger: dschroedinger, dschroedinger_dynamic
using ...stochastic
import ...stochastic.stochastic_master: dmaster_stochastic, dmaster_stochastic_nl, dmaster_stoch_dynamic, dmaster_stoch_dynamic_nl, dlindblad

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

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
* `noise_processes=0`: Number of distinct white-noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch_quantum`. Add 1 if `fstoch_classical` is specificed.
        If unset, the number is automatically calculated from function outputs.
        NOTE: Set this number if you want to avoid an initial calculation of
        function outputs!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function schroedinger_semiclassical(tspan, state0::State{Ket}, fquantum::Function,
                fclassical::Function; fstoch_quantum::Union{Void, Function}=nothing,
                fstoch_classical::Union{Void, Function}=nothing,
                fout::Union{Function,Void}=nothing,
                noise_processes::Int=0,
                kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    dschroedinger_det(t::Float64, state::State{Ket}, dstate::State{Ket}) = semiclassical.dschroedinger_dynamic(t, state, fquantum, fclassical, dstate)

    if isa(fstoch_quantum, Void) && isa(fstoch_classical, Void)
        throw(ArgumentError("No stochastic functions provided!"))
    end

    x0 = Vector{Complex128}(length(state0))
    recast!(state0, x0)
    state = copy(state0)
    dstate = copy(state0)

    if noise_processes == 0
        n = 0
        if isa(fstoch_quantum, Function)
            fs_out = fstoch_quantum(0.0, state0.quantum, state0.classical)
            n += length(fs_out)
        end
        if isa(fstoch_classical, Function)
            n += 1
        end
    else
        n = noise_processes
    end

    dschroedinger_stoch(t::Float64, state::State{Ket}, dstate::State{Ket}, index::Int) =
        dschroedinger_stochastic(t, state, fstoch_quantum, fstoch_classical, dstate, index)
    integrate_stoch(tspan_, dschroedinger_det, dschroedinger_stoch, x0, state, dstate, fout, n; kwargs...)
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
* `fstoch_quantum=nothing`: Function `f(t, rho, u) -> Js, Jsdagger` or
        `f(t, psi, u) -> Js, Jsdagger, rates_s` that returns the stochastic
        operator for the superoperator of the form `Js[i]*rho + rho*Jsdagger[i]`.
* `fstoch_classical=nothing`: Function `f(t, rho, u, du)` that calculates the
        stochastic terms of the derivative `du`.
* `fstoch_H=nothing`: Function `f(t, rho, u) -> Hs` providing a vector of operators
        that correspond to stochastic terms of the Hamiltonian.
* `fstoch_J=nothing`: Function `f(t, rho, u) -> (J, Jdagger)` or
        `f(t, rho, u) -> (J, Jdagger, rates)` giving a stochastic
        Lindblad term.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `rates_s=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the stochastic jump operators. If nothing is specified all rates are assumed
        to be 1.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `noise_processes=0`: Number of distinct white-noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by all stochastic functions. Add 1 for classical noise.
        If unset, the number is calculated automatically from the function outputs.
        NOTE: Set this number if you want to avoid an initial calculation of
        function outputs!
* `nonlinear=true`: Specify whether or not to include the nonlinear term
        `expect(Js[i] + Jsdagger[i],rho)*rho` in the equation. This ensures
        the trace of `rho` is conserved.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_semiclassical(tspan::Vector{Float64}, rho0::State{DenseOperator},
                fquantum::Function, fclassical::Function;
                fstoch_quantum::Union{Function, Void}=nothing,
                fstoch_classical::Union{Function, Void}=nothing,
                fstoch_H::Union{Function, Void}=nothing, fstoch_J::Union{Function, Void}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                noise_processes::Int=0, nonlinear::Bool=true,
                kwargs...)

    tmp = copy(rho0.quantum)

    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to set them as ones or use diagonaljumps."))
    end
    if isa(fstoch_quantum, Void) && isa(fstoch_classical, Void) && isa(fstoch_H, Void) && isa(fstoch_J, Void)
        throw(ArgumentError("No stochastic functions provided!"))
    end

    if noise_processes == 0
        n = 0
        if isa(fstoch_quantum, Function)
            fq_out = fstoch_quantum(0, rho0.quantum, rho0.classical)
            n += length(fq_out[1])
        end
        if isa(fstoch_classical, Function)
            n += 1
        end
        if isa(fstoch_H, Function)
            n += length(fstoch_H(0, rho0.quantum, rho0.classical))
        end
        if isa(fstoch_J, Function)
            n += length(fstoch_J(0, rho0.quantum, rho0.classical)[1])
        end
    else
        n = noise_processes
    end

    dmaster_determ(t::Float64, rho::State{DenseOperator}, drho::State{DenseOperator}) =
            dmaster_h_dynamic(t, rho, fquantum, fclassical, rates, drho, tmp)
    if isa(fstoch_H, Void) && isa(fstoch_J, Void)
        if nonlinear
            dmaster_stoch_std_nl(t::Float64, rho::State{DenseOperator},
                            drho::State{DenseOperator}, index::Int) =
                dmaster_stoch_dynamic_nl(t, rho, fstoch_quantum, fstoch_classical,
                            rates_s, drho, index)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std_nl, rho0, fout, n; kwargs...)
        else
            dmaster_stoch_std(t::Float64, rho::State{DenseOperator},
                            drho::State{DenseOperator}, index::Int) =
                dmaster_stoch_dynamic(t, rho, fstoch_quantum, fstoch_classical,
                            rates_s, drho, index)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std, rho0, fout, n; kwargs...)
        end
    else
        if nonlinear
            dmaster_stoch_gen_nl(t::Float64, rho::State{DenseOperator},
                            drho::State{DenseOperator}, index::Int) =
                dmaster_stoch_dynamic_general_nl(t, rho, fstoch_quantum,
                            fstoch_classical, fstoch_H, fstoch_J, rates, rates_s,
                            drho, tmp, index)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen_nl, rho0, fout, n; kwargs...)
        else
            dmaster_stoch_gen(t::Float64, rho::State{DenseOperator},
                            drho::State{DenseOperator}, index::Int) =
                dmaster_stoch_dynamic_general(t, rho, fstoch_quantum,
                            fstoch_classical, fstoch_H, fstoch_J, rates, rates_s,
                            drho, tmp, index)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen, rho0, fout, n; kwargs...)
        end
    end
end
master_semiclassical(tspan::Vector{Float64}, psi0::State{Ket}, args...; kwargs...) =
        master_semiclassical(tspan, dm(psi0), args...; kwargs...)

function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Function,
            fstoch_classical::Function, dstate::State{Ket}, index::Int)
    H = fstoch_quantum(t, state.quantum, state.classical)
    if index <= length(H)
        dschroedinger(state.quantum, H[index], dstate.quantum)
    else
        fstoch_classical(t, state.quantum, state.classical, dstate.classical)
    end
end
function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Function,
            fstoch_classical::Void, dstate::State{Ket}, index::Int)
    H = fstoch_quantum(t, state.quantum, state.classical)
    dschroedinger(state.quantum, H[index], dstate.quantum)
end
function dschroedinger_stochastic(t::Float64, state::State{Ket}, fstoch_quantum::Void,
            fstoch_classical::Function, dstate::State{Ket}, index::Int)
    fstoch_classical(t, state.quantum, state.classical, dstate.classical)
end

function dmaster_stoch_dynamic(t::Float64, state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Void,
            rates_s::DecayRates, dstate::State{DenseOperator}, index::Int)
    fstoch_quantum_(t, rho) = fstoch_quantum(t, state.quantum, state.classical)
    dmaster_stoch_dynamic(t, state.quantum, fstoch_quantum_, rates_s,
                dstate.quantum, index)
end
function dmaster_stoch_dynamic(t::Float64, state::State{DenseOperator}, fstoch_quantum::Void,
            fstoch_classical::Function,
            rates_s::DecayRates, dstate::State{DenseOperator}, index::Int)
    fstoch_classical(t, state.quantum, state.classical, dstate.classical)
end
function dmaster_stoch_dynamic(t::Float64, state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Function,
            rates_s::DecayRates, dstate::State{DenseOperator}, index::Int)
    result = fstoch_quantum(t, state.quantum, state.classical)
    if index <= length(result[1])
        @assert 2 <= length(result) <= 3
        if length(result) == 2
            Js, Jsdagger = result
            rates_s_ = rates_s
        else
            Js, Jsdagger, rates_s_ = result
        end
        dmaster_stochastic(state.quantum, nothing, rates_s_, Js, Jsdagger, dstate.quantum, index)
    else
        fstoch_classical(t, state.quantum, state.classical, dstate.classical)
    end
end

function dmaster_stoch_dynamic_nl(t::Float64, state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Void,
            rates_s::DecayRates, dstate::State{DenseOperator}, index::Int)
    fstoch_quantum_(t, rho) = fstoch_quantum(t, state.quantum, state.classical)
    dmaster_stoch_dynamic_nl(t, state.quantum, fstoch_quantum_, rates_s,
                dstate.quantum, index)
end
function dmaster_stoch_dynamic_nl(t::Float64, state::State{DenseOperator}, fstoch_quantum::Void,
            fstoch_classical::Function, args...)
    dmaster_stoch_dynamic(t, state, fstoch_quantum, fstoch_classical, args...)
end
function dmaster_stoch_dynamic_nl(t::Float64, state::State{DenseOperator}, fstoch_quantum::Function,
            fstoch_classical::Function,
            rates_s::DecayRates, dstate::State{DenseOperator}, index::Int)
    result = fstoch_quantum(t, state.quantum, state.classical)
    if index <= length(result[1])
        @assert 2 <= length(result) <= 3
        if length(result) == 2
            Js, Jsdagger = result
            rates_s_ = rates_s
        else
            Js, Jsdagger, rates_s_ = result
        end
        dmaster_stochastic(state.quantum, nothing, rates_s_, Js, Jsdagger, dstate.quantum, index)
        if isa(rates_s_, Void)
            dstate.quantum.data .-= (expect(Js[index], state.quantum) + expect(Jsdagger[index], state.quantum))*state.quantum.data
        else
            dstate.quantum.data .-= rates_s_[index]*(expect(Js[index], state.quantum) + expect(Jsdagger[index], state.quantum))*state.quantum.data
        end
    else
        fstoch_classical(t, state.quantum, state.classical, dstate.classical)
    end
end


function dmaster_stoch_dynamic_general(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, state.quantum, state.classical)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], state.quantum, 0.0, dstate.quantum)
        operators.gemm!(1.0im, state.quantum, H[index], 1.0, dstate.quantum)
    else
        dmaster_stoch_dynamic(t, state, fstoch_quantum, fstoch_classical,
                rates_s, dstate, index-length(H))
    end
end
function dmaster_stoch_dynamic_general(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    result_J = fstoch_J(t, state.quantum, state.classical)
    if index <= length(result_J[1])
        @assert 2 <= length(result_J) <= 3
        if length(result_J) == 2
            J, Jdagger = result_J
            rates_ = rates
        else
            J, Jdagger, rates_ = result_J
        end
        dlindblad(state.quantum, rates_, J, Jdagger, dstate.quantum, tmp, index)
    else
        dmaster_stoch_dynamic(t, state, fstoch_quantum, fstoch_classical,
                rates_s, dstate, index-length(result_J[1]))
    end
end
function dmaster_stoch_dynamic_general(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, state.quantum, state.classical)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], state.quantum, 0.0, dstate.quantum)
        operators.gemm!(1.0im, state.quantum, H[index], 1.0, dstate.quantum)
    else
        dmaster_stoch_dynamic_general(t, state, fstoch_quantum, fstoch_classical,
                nothing, fstoch_J, rates, rates_s, dstate, tmp, index-length(H))
    end
end

function dmaster_stoch_dynamic_general_nl(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, state.quantum, state.classical)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], state.quantum, 0.0, dstate.quantum)
        operators.gemm!(1.0im, state.quantum, H[index], 1.0, dstate.quantum)
    else
        dmaster_stoch_dynamic_nl(t, state, fstoch_quantum, fstoch_classical,
                rates_s, dstate, index-length(H))
    end
end
function dmaster_stoch_dynamic_general_nl(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    result_J = fstoch_J(t, state.quantum, state.classical)
    if index <= length(result_J[1])
        @assert 2 <= length(result_J) <= 3
        if length(result_J) == 2
            J, Jdagger = result_J
            rates_ = rates
        else
            J, Jdagger, rates_ = result_J
        end
        dlindblad(state.quantum, rates_, J, Jdagger, dstate.quantum, tmp, index)
    else
        dmaster_stoch_dynamic_nl(t, state, fstoch_quantum, fstoch_classical,
                rates_s, dstate, index-length(result_J[1]))
    end
end
function dmaster_stoch_dynamic_general_nl(t::Float64, state::State{DenseOperator},
            fstoch_quantum::Union{Function, Void}, fstoch_classical::Union{Function, Void},
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            dstate::State{DenseOperator}, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, state.quantum, state.classical)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], state.quantum, 0.0, dstate.quantum)
        operators.gemm!(1.0im, state.quantum, H[index], 1.0, dstate.quantum)
    else
        dmaster_stoch_dynamic_general_nl(t, state, fstoch_quantum, fstoch_classical,
                nothing, fstoch_J, rates, rates_s, dstate, tmp, index-length(H))
    end
end

function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::State{DenseOperator}, fout::Union{Void, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = Vector{Complex128}(length(rho0))
    recast!(rho0, x0)
    state = copy(rho0)
    dstate = copy(rho0)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function recast!(state::State, x::SubArray{Complex128, 1})
    N = length(state.quantum)
    copy!(x, 1, state.quantum.data, 1, N)
    copy!(x, N+1, state.classical, 1, length(state.classical))
    x
end
function recast!(x::SubArray{Complex128, 1}, state::State)
    N = length(state.quantum)
    copy!(state.quantum.data, 1, x, 1, N)
    copy!(state.classical, 1, x, N+1, length(state.classical))
end

end # module
