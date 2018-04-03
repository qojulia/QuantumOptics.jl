module stochastic_master

export master, master_dynamic

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ...timeevolution
import ...timeevolution: integrate_stoch, recast!
import ...timeevolution.timeevolution_master: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}
const DiffArray = Union{Vector{Complex128}, Array{Complex128, 2}}

"""
    stochastic.master(tspan, rho0, H, J, Js; <keyword arguments>)

Time-evolution according to a stochastic master equation.

For dense arguments the `master` function calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Deterministic part of the Hamiltonian.
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
* `nonlinear=true`: Specify whether or not to include the nonlinear term
        `expect(Js[i] + Jsdagger[i],rho)*rho` in the equation. This ensures
        the trace of `rho` is conserved.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master(tspan, rho0::DenseOperator, H::Operator,
                J::Vector, Js::Vector; Hs::Union{Void, Vector}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                Jdagger::Vector=dagger.(J), Jsdagger::Vector=dagger.(Js),
                fout::Union{Function,Void}=nothing,
                nonlinear::Bool=true,
                kwargs...)

    tmp = copy(rho0)

    if isa(rates_s, Matrix{Float64})
        throw(ArgumentError("A matrix of stochastic rates is ambiguous! Please provide a vector of stochastic rates.
        You may want to use diagonaljumps."))
    end

    n = length(Js) + (isa(Hs, Void) ? 0 : length(Hs))

    if nonlinear
        dmaster_stoch_nl(dx::DiffArray,
                t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
            dmaster_stochastic_nl(dx, rho, Hs, rates_s, Js, Jsdagger, drho, n)
    else
        dmaster_stoch_lin(dx::DiffArray,
                t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
            dmaster_stochastic(dx, rho, Hs, rates_s, Js, Jsdagger, drho, n)
    end

    isreducible = check_master(rho0, H, J, Jdagger, rates)
    check_master(rho0, H, Js, Jsdagger, rates_s)
    if !isreducible
        dmaster_h_determ(t::Float64, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        if nonlinear
            integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch_nl, rho0, fout, n; kwargs...)
        else
            integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch_lin, rho0, fout, n; kwargs...)
        end
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
        if nonlinear
            integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch_nl, rho0, fout, n; kwargs...)
        else
            integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch_lin, rho0, fout, n; kwargs...)
        end
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
* `nonlinear=true`: Specify whether or not to include the nonlinear term
        `expect(Js[i] + Jsdagger[i],rho)*rho` in the equation. This ensures
        the trace of `rho` is conserved.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan::Vector{Float64}, rho0::DenseOperator, fdeterm::Function, fstoch::Function;
                fstoch_H::Union{Function, Void}=nothing, fstoch_J::Union{Function, Void}=nothing,
                rates::DecayRates=nothing, rates_s::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                noise_processes::Int=0, nonlinear::Bool=true,
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
        if nonlinear
            dmaster_stoch_std_nl(dx::DiffArray,
                    t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
                dmaster_stoch_dynamic_nl(dx, t, rho, fstoch, rates_s, drho, n)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std_nl, rho0, fout, n; kwargs...)
        else
            dmaster_stoch_std(dx::DiffArray,
                    t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
                dmaster_stoch_dynamic(dx, t, rho, fstoch, rates_s, drho, n)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_std, rho0, fout, n; kwargs...)
        end
    else
        if nonlinear
            dmaster_stoch_gen_nl(dx::DiffArray,
                    t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
                dmaster_stoch_dynamic_general_nl(dx, t, rho, fstoch, fstoch_H, fstoch_J,
                        rates, rates_s, drho, tmp, n)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen_nl, rho0, fout, n; kwargs...)
        else
            dmaster_stoch_gen(dx::DiffArray,
                    t::Float64, rho::DenseOperator, drho::DenseOperator, n::Int) =
                dmaster_stoch_dynamic_general(dx, t, rho, fstoch, fstoch_H, fstoch_J,
                        rates, rates_s, drho, tmp, n)
            integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch_gen, rho0, fout, n; kwargs...)
        end
    end
end
master_dynamic(tspan::Vector{Float64}, psi0::Ket, args...; kwargs...) = master_dynamic(tspan, dm(psi0), args...; kwargs...)

# Terms in SME
function dneumann(rho::DenseOperator, H::Operator, drho::DenseOperator)
    operators.gemm!(-1.0im, H, rho, 0.0, drho)
    operators.gemm!(1.0im, rho, H, 1.0, drho)
end

function dlindblad(rho::DenseOperator, rates::Void, J::Vector, Jdagger::Vector,
        drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(1, J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
end
function dlindblad(rho::DenseOperator, rates::Vector{Float64}, J::Vector,
        Jdagger::Vector, drho::DenseOperator, tmp::DenseOperator, i::Int)
    operators.gemm!(rates[i], J[i], rho, 0, tmp)
    operators.gemm!(1, tmp, Jdagger[i], 1, drho)
end

function dwiseman(rho::DenseOperator, rates::Void, J::Vector,
        Jdagger::Vector, drho::DenseOperator, i::Int)
    operators.gemm!(1, J[i], rho, 0, drho)
    operators.gemm!(1, rho, Jdagger[i], 1, drho)
end
function dwiseman(rho::DenseOperator, rates::Vector{Float64}, J::Vector,
        Jdagger::Vector, drho::DenseOperator, i::Int)
    operators.gemm!(rates[i], J[i], rho, 0, drho)
    operators.gemm!(rates[i], rho, Jdagger[i], 1, drho)
end

function dwiseman_nl(rho::DenseOperator, rates::Void, J::Vector,
        Jdagger::Vector, drho::DenseOperator, i::Int)
    operators.gemm!(1, J[i], rho, 0, drho)
    operators.gemm!(1, rho, Jdagger[i], 1, drho)
    drho.data .-= (expect(J[i], rho) + expect(Jdagger[i], rho))*rho.data
end
function dwiseman_nl(rho::DenseOperator, rates::Vector{Float64}, J::Vector,
        Jdagger::Vector, drho::DenseOperator, i::Int)
    operators.gemm!(rates[i], J[i], rho, 0, drho)
    operators.gemm!(rates[i], rho, Jdagger[i], 1, drho)
    drho.data .-= rates[i]*(expect(J[i], rho) + expect(Jdagger[i], rho))*rho.data
end

# Derivative functions
function dmaster_stochastic(dx::Vector{Complex128}, rho::DenseOperator, H::Void, rates::DecayRates,
            J::Vector, Jdagger::Vector, drho::DenseOperator, ::Int)
    recast!(dx, drho)
    dwiseman(rho, rates, J, Jdagger, drho, 1)
    recast!(drho, dx)
end
function dmaster_stochastic(dx::Array{Complex128, 2}, rho::DenseOperator, H::Void, rates::DecayRates,
            J::Vector, Jdagger::Vector, drho::DenseOperator, n::Int)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dwiseman(rho, rates, J, Jdagger, drho, i)
        recast!(drho, dx_i)
    end
end
function dmaster_stochastic(dx::Array{Complex128, 2}, rho::DenseOperator, H::Vector,
            rates::DecayRates, J::Vector, Jdagger::Vector, drho::DenseOperator, n::Int)
    m = length(H)
    for i=n-m+1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dneumann(rho, H[i-n+m], drho)
        recast!(drho, dx_i)
    end
    dmaster_stochastic(dx, rho, nothing, rates, J, Jdagger, drho, n-m)
end

function dmaster_stochastic_nl(dx::Vector{Complex128}, rho::DenseOperator, H::Void, rates::DecayRates,
            J::Vector, Jdagger::Vector, drho::DenseOperator, ::Int)
    recast!(dx, drho)
    dwiseman_nl(rho, rates, J, Jdagger, drho, 1)
    recast!(drho, dx)
end
function dmaster_stochastic_nl(dx::Array{Complex128, 2}, rho::DenseOperator, H::Void, rates::DecayRates,
            J::Vector, Jdagger::Vector, drho::DenseOperator, n::Int)
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dwiseman_nl(rho, rates, J, Jdagger, drho, i)
        recast!(drho, dx_i)
    end
end
function dmaster_stochastic_nl(dx::Array{Complex128, 2}, rho::DenseOperator, H::Vector,
            rates::DecayRates, J::Vector, Jdagger::Vector, drho::DenseOperator, n::Int)
    m = length(H)
    for i=n-m+1:m
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dneumann(rho, H[i-n+m], drho)
        recast!(drho, dx_i)
    end
    dmaster_stochastic_nl(dx, rho, nothing, rates, J, Jdagger, drho, n-m)
end

function dmaster_stoch_dynamic(dx::DiffArray, t::Float64, rho::DenseOperator,
            f::Function, rates::DecayRates, drho::DenseOperator, n::Int)
    result = f(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic(dx, rho, nothing, rates_, J, Jdagger, drho, n)
end
function dmaster_stoch_dynamic_nl(dx::DiffArray, t::Float64, rho::DenseOperator,
            f::Function, rates::DecayRates, drho::DenseOperator, n::Int)
    result = f(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic_nl(dx, rho, nothing, rates_, J, Jdagger, drho, n)
end

function dmaster_stoch_dynamic_general(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    H = fstoch_H(t, rho)
    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates_s
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic(dx, rho, H, rates_, J, Jdagger, drho, n)
end
function dmaster_stoch_dynamic_general(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    result_J = fstoch_J(t, rho)
    @assert 2 <= length(result_J) <= 3
    if length(result_J) == 2
        J_stoch, J_stoch_dagger = result_J
        rates_ = rates
    else
        J_stoch, J_stoch_dagger, rates_ = result_J
    end
    l = length(J_stoch)

    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_s_ = rates_s
    else
        J, Jdagger, rates_s_ = result
    end

    for i=n-l+1:n-l
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, i-n+l)
        recast!(drho, dx_i)
    end
    dmaster_stochastic(dx, rho, nothing, rates_s_, J, Jdagger, drho, n-l)
end
function dmaster_stoch_dynamic_general(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    H = fstoch_H(t, rho)

    result_J = fstoch_J(t, rho)
    @assert 2 <= length(result_J) <= 3
    if length(result_J) == 2
        J_stoch, J_stoch_dagger = result_J
        rates_ = rates
    else
        J_stoch, J_stoch_dagger, rates_ = result_J
    end
    l = length(J_stoch)

    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_s_ = rates_s
    else
        J, Jdagger, rates_s_ = result
    end

    for i=n-l+1:n-l
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, i-n+l)
        recast!(drho, dx_i)
    end
    dmaster_stochastic(dx, rho, H, rates_s_, J, Jdagger, drho, n-l)
end

function dmaster_stoch_dynamic_general_nl(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Void, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    H = fstoch_H(t, rho)
    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_ = rates_s
    else
        J, Jdagger, rates_ = result
    end
    dmaster_stochastic_nl(dx, rho, H, rates_, J, Jdagger, drho, n)
end
function dmaster_stoch_dynamic_general_nl(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Void, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    result_J = fstoch_J(t, rho)
    @assert 2 <= length(result_J) <= 3
    if length(result_J) == 2
        J_stoch, J_stoch_dagger = result_J
        rates_ = rates
    else
        J_stoch, J_stoch_dagger, rates_ = result_J
    end
    l = length(J_stoch)

    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_s_ = rates_s
    else
        J, Jdagger, rates_s_ = result
    end

    for i=n-l+1:n-l
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, i-n+l)
        recast!(drho, dx_i)
    end
    dmaster_stochastic_nl(dx, rho, nothing, rates_s_, J, Jdagger, drho, n-l)
end
function dmaster_stoch_dynamic_general_nl(dx::Array{Complex128, 2}, t::Float64, rho::DenseOperator, fstoch::Function,
            fstoch_H::Function, fstoch_J::Function, rates::DecayRates, rates_s::DecayRates,
            drho::DenseOperator, tmp::DenseOperator, n::Int)
    H = fstoch_H(t, rho)

    result_J = fstoch_J(t, rho)
    @assert 2 <= length(result_J) <= 3
    if length(result_J) == 2
        J_stoch, J_stoch_dagger = result_J
        rates_ = rates
    else
        J_stoch, J_stoch_dagger, rates_ = result_J
    end
    l = length(J_stoch)

    result = fstoch(t, rho)
    @assert 2 <= length(result) <= 3
    if length(result) == 2
        J, Jdagger = result
        rates_s_ = rates_s
    else
        J, Jdagger, rates_s_ = result
    end

    for i=n-l+1:n-l
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        dlindblad(rho, rates_, J_stoch, J_stoch_dagger, drho, tmp, i-n+l)
        recast!(drho, dx_i)
    end
    dmaster_stochastic_nl(dx, rho, H, rates_s_, J, Jdagger, drho, n-l)
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
