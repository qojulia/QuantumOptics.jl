module stochastic_master

export master, master_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...timeevolution
import ...timeevolution: integrate_stoch, recast!
import ...timeevolution.timeevolution_master: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

"""
    stochastic.master(tspan, rho0, H, Hs, J; <keyword arguments>)

Time-evolution according to a stochastic master equation.

For dense arguments the `master` function calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Deterministic part of the Hamiltonian.
* `Hs`: Operator or vector of operators specifying the stochastic part of the
        Hamiltonian.
* `J`: Vector containing all deterministic
        jump operators which can be of any arbitrary operator type.
* `Js`: Vector containing the stochastic jump operators for a superoperator
        describing a measurement which has the form of the standard linear
        stochastic master equation, `Js[i]*rho + rho*Jsdagger[i]`.
* `Hs=nothing`: Vector containing additional stochastic terms of the Hamiltonian.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `rates_s=nothing`: Vector specifying the coefficients (decay rates)
        for the stochastic jump operators. If nothing is specified all rates
        are assumed to be 1.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
        operators. If they are not given they are calculated automatically.
* `Jsdagger=dagger.(Js)`: Vector containing the hermitian conjugates of the
        stochastic jump operators.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master(tspan, rho0::DenseOperator, H::Operator,
                J::Vector, Js::Vector; Hs::Union{Void, Vector}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                Jdagger::Vector=dagger.(J), Jsdagger::Vector=dagger.(Js),
                fout::Union{Function,Void}=nothing,
                kwargs...)

    tmp = copy(rho0)

    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    n = length(Js) + (isa(Hs, Void) ? 0 : length(Hs))
    dmaster_stoch(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) = dmaster_stochastic(rho, Hs, rates_s, Js, Jsdagger, drho, tmp, index)

    isreducible = check_master(rho0, H, J, Jdagger, rates)
    check_master(rho0, H, Js, Jsdagger, rates_s)
    if !isreducible
        dmaster_h_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    else
        Hnh = copy(H)
        if typeof(rates) == Matrix{Float64}
            for i=1:length(J), j=1:length(J)
                Hnh -= 0.5im*rates[i,j]*Jdagger[i]*J[j]
            end
        elseif typeof(rates) == Vector{Float64}
            for i=1:length(J)
                Hnh -= 0.5im*rates[i]*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= 0.5im*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)

        dmaster_nh_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    end
end
master(tspan, psi0::Ket, args...; kwargs...) = master(tspan, dm(psi0), args...; kwargs...)

"""
    stochastic.master_dynamic(tspan, rho0, f, fs; <keyword arguments>)

Time-evolution according to a stochastic master equation with a
dynamic Hamiltonian and J.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `fdeterm`: Function `f(t, rho) -> (H, J, Jdagger)` or
        `f(t, rho) -> (H, J, Jdagger, rates)` giving the deterministic
        part of the master equation.
* `fstoch`: Function `f(t, rho) -> (Js, Jsdagger)` or
        `f(t, rho) -> (Js, Jsdagger, rates)` giving the stochastic superoperator
        of the form `Js[i]*rho + rho*Jsdagger[i]`.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `rates_s=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the stochastic jump operators. If nothing is specified all rates are assumed
        to be 1.
* `fstoch_H=nothing`: Function `f(t, rho) -> Hs` providing a vector of operators
        that correspond to stochastic terms of the Hamiltonian.
* `fstoch_J=nothing`: Function `f(t, rho) -> (J, Jdagger)` or
        `f(t, rho) -> (J, Jdagger, rates)` giving a stochastic
        Lindblad term.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `noise_processes=0`: Number of distinct white-noise processes in the equation.
        This number has to be equal to the total number of noise operators
        returned by `fstoch` and all optional functions. If unset, the number
        is calculated automatically from the function outputs. NOTE: Set this
        number if you want to avoid an initial calculation of function outputs!
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan::Vector{Float64}, rho0::DenseOperator, fdeterm::Function, fstoch::Function;
                fstoch_H::Union{Function, Void}=nothing, fstoch_J::Union{Function, Void}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                noise_processes::Int=0,
                kwargs...)

    tmp = copy(rho0)

    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    if noise_processes == 0
        fs_out = fstoch(0, rho0)
        n = length(fs_out[1])
        if isa(fstoch_H, Function)
            n += length(fstoch_H(0, rho0))
        end
        if isa(fstoch_J, Function)
            n += length(fstoch_J(0, rho0)[1])
        end
    else
        n = noise_processes
    end

    dmaster_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h_dynamic(t, rho, fdeterm, rates, drho, tmp)
    if isa(fstoch_H, Void) && isa(fstoch_J, Void)
        dmaster_stoch_std(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
            dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index)

        integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std, rho0, fout, n; kwargs...)
    else
        dmaster_stoch_gen(t::Float64, rho::DenseOperator, drho::DenseOperator, index::Int) =
            dmaster_stoch_dynamic_general(t, rho, fstoch, fstoch_H, fstoch_J,
                    rates, rates_s, drho, tmp, index)

        integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen, rho0, fout, n; kwargs...)
    end
end
master_dynamic(tspan::Vector{Float64}, psi0::Ket, args...; kwargs...) = master_dynamic(tspan, dm(psi0), args...; kwargs...)

function dmaster_stochastic(rho::DenseOperator, H::Void, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    operators.gemm!(1, J[index], rho, 0, drho)
    operators.gemm!(1, rho, Jdagger[index], 1, drho)
    return drho
end
function dmaster_stochastic(rho::DenseOperator, H::Void, rates::Vector{Float64},
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    operators.gemm!(rates[index], J[index], rho, 0, drho)
    operators.gemm!(rates[index], rho, Jdagger[index], 1, drho)
    return drho
end

function dmaster_stochastic(rho::DenseOperator, H::Vector, rates::Void,
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(1, J[index], rho, 0, drho)
        operators.gemm!(1, rho, Jdagger[index], 1, drho)
    end
    return drho
end
function dmaster_stochastic(rho::DenseOperator, H::Vector, rates::Vector{Float64},
            J::Vector, Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator,
            index::Int)
    if index > length(J)
        operators.gemm!(-1.0im, H[index - length(J)], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index - length(J)], 1.0, drho)
    else
        operators.gemm!(rates[index], J[index], rho, 0, drho)
        operators.gemm!(rates[index], rho, Jdagger[index], 1, drho)
    end
    return drho
end

function dmaster_stoch_dynamic(t::Float64, rho::DenseOperator, f::Function, rates::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    result = f(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic(rho, nothing, rates_, J, Jdagger, drho, tmp, index)
end

function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, rho)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index], 1.0, drho)
    else
        dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index-length(H))
    end
end
function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    result_J = fstoch_J(t, rho)
    if index <= length(result_J[1])
        @assert 2 <= length(result_J) <= 3
        if length(result_J) == 2
            J_stoch, J_stoch_dagger = result_J
            rates_ = rates
        else
            J_stoch, J_stoch_dagger, rates_ = result_J
        end
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, index)
    else
        dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index-length(result_J[1]))
    end
end
function dmaster_stoch_dynamic_general(t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, index::Int)
    H = fstoch_H(t, rho)
    result_J = fstoch_J(t, rho)
    if index <= length(H)
        operators.gemm!(-1.0im, H[index], rho, 0.0, drho)
        operators.gemm!(1.0im, rho, H[index], 1.0, drho)
    elseif length(H) < index <= length(H) + length(result_J[1])
        @assert 2 <= length(result_J) <= 3
        if length(result_J) == 2
            J_stoch, J_stoch_dagger = result_J
            rates_ = rates
        else
            J_stoch, J_stoch_dagger, rates_ = result_J
        end
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, index-length(H))
    else
        dmaster_stoch_dynamic(t, rho, fstoch, rates_s, drho, tmp, index-length(H)-length(result_J[1]))
    end
end

function dlindblad(rho::DenseOperator, rates::Void, J::Vector, Jdagger::Vector,
    drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(1, J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    return drho
end
function dlindblad(rho::DenseOperator, rates::Vector{Float64}, J::Vector,
    Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(rates[i], J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    return drho
end

function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::DenseOperator, fout::Union{Void, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = reshape(rho0.data, length(rho0))
    state = copy(rho0)
    dstate = copy(rho0)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function recast!(x::SubArray{Complex128, 1}, rho::DenseOperator)
    rho.data = reshape(x, size(rho.data))
end
recast!(state::DenseOperator, x::SubArray{Complex128, 1}) = (x[:] = state.data)

end # module
