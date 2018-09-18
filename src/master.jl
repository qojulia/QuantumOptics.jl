module timeevolution_master

export master, master_nh, master_h, master_dynamic, master_nh_dynamic

import ..integrate, ..recast!, ..QO_CHECKS

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using SparseArrays, LinearAlgebra


const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}
const OperatorDataType = Union{Matrix{ComplexF64},SparseMatrixCSC{ComplexF64,Int}}

"""
    timeevolution.master_h(tspan, rho0, H, J; <keyword arguments>)

Integrate the master equation with dmaster_h as derivative function.

Further information can be found at [`master`](@ref).
"""
function master_h(tspan, rho0::T1, H::AbstractOperator{B,B}, J::Vector{T2};
                rates::DecayRates=nothing,
                Jdagger::Vector{T2}=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B}}
    check_master(rho0, H, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::T1, drho::T1) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end

"""
    timeevolution.master_nh(tspan, rho0, H, J; <keyword arguments>)

Integrate the master equation with dmaster_nh as derivative function.

In this case the given Hamiltonian is assumed to be the non-hermitian version:
```math
H_{nh} = H - \\frac{i}{2} \\sum_k J^†_k J_k
```
Further information can be found at [`master`](@ref).
"""
function master_nh(tspan, rho0::T1, Hnh::T2, J::Vector{T3};
                rates::DecayRates=nothing,
                Hnhdagger::T2=dagger(Hnh),
                Jdagger::Vector{T3}=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B},T3<:AbstractOperator{B,B}}
    check_master(rho0, Hnh, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::T1, drho::T1) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end

"""
    timeevolution.master(tspan, rho0, H, J; <keyword arguments>)

Time-evolution according to a master equation.

There are two implementations for integrating the master equation:

* [`master_h`](@ref): Usual formulation of the master equation.
* [`master_nh`](@ref): Variant with non-hermitian Hamiltonian.

For dense arguments the `master` function calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Arbitrary operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
        operator type.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
        operators. If they are not given they are calculated automatically.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master(tspan, rho0::T1, H::AbstractOperator{B,B}, J::Vector{T2};
                rates::DecayRates=nothing,
                Jdagger::Vector{T2}=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B}}
    isreducible = check_master(rho0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(rho0)
        dmaster_h_(t, rho::T1, drho::T1) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        return integrate_master(tspan, dmaster_h_, rho0, fout; kwargs...)
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
        tmp = copy(rho0)
        dmaster_nh_(t, rho::T1, drho::T1) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        return integrate_master(tspan, dmaster_nh_, rho0, fout; kwargs...)
    end
end

"""
    timeevolution.master_dynamic(tspan, rho0, f; <keyword arguments>)

Time-evolution according to a master equation with a dynamic non-hermitian Hamiltonian and J.

In this case the given Hamiltonian is assumed to be the non-hermitian version.
```math
H_{nh} = H - \\frac{i}{2} \\sum_k J^†_k J_k
```
The given function can either be of the form `f(t, rho) -> (Hnh, Hnhdagger, J, Jdagger)`
or `f(t, rho) -> (Hnh, Hnhdagger, J, Jdagger, rates)` For further information look
at [`master_dynamic`](@ref).
"""
function master_nh_dynamic(tspan, rho0::Operator{B,B}, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where B<:Basis
    tmp = copy(rho0)
    dmaster_(t, rho::Operator{B,B}, drho::Operator{B,B}) = dmaster_nh_dynamic(t, rho, f, rates, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end

"""
    timeevolution.master_dynamic(tspan, rho0, f; <keyword arguments>)

Time-evolution according to a master equation with a dynamic Hamiltonian and J.

There are two implementations for integrating the master equation with dynamic
operators:

* [`master_dynamic`](@ref): Usual formulation of the master equation.
* [`master_nh_dynamic`](@ref): Variant with non-hermitian Hamiltonian.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `f`: Function `f(t, rho) -> (H, J, Jdagger)` or `f(t, rho) -> (H, J, Jdagger, rates)`
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_dynamic(tspan, rho0::Operator{B,B}, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where B<:Basis
    tmp = copy(rho0)
    dmaster_(t, rho::Operator{B,B}, drho::Operator{B,B}) = dmaster_h_dynamic(t, rho, f, rates, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end


# Automatically convert Ket states to density operators
master(tspan, psi0::Ket, H::AbstractOperator, J::Vector; kwargs...) = master(tspan, dm(psi0), H, J; kwargs...)
master_h(tspan, psi0::Ket, H::AbstractOperator, J::Vector; kwargs...) = master_h(tspan, dm(psi0), H, J; kwargs...)
master_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J::Vector; kwargs...) = master_nh(tspan, dm(psi0), Hnh, J; kwargs...)
master_dynamic(tspan, psi0::Ket, f::Function; kwargs...) = master_dynamic(tspan, dm(psi0), f; kwargs...)
master_nh_dynamic(tspan, psi0::Ket, f::Function; kwargs...) = master_nh_dynamic(tspan, dm(psi0), f; kwargs...)


# Recasting needed for the ODE solver is just providing the underlying data
# TODO: recast! for sparse rho
function recast!(x::Array{ComplexF64, 2}, rho::Operator{B,B,T}) where {B<:Basis,T<:Matrix{ComplexF64}}
    rho.data = x
end
recast!(rho::Operator{B,B}, x::Array{ComplexF64, 2}) where B<:Basis = nothing

function integrate_master(tspan, df::Function, rho0::Operator{B,B},
                        fout::Union{Nothing, Function}; kwargs...) where B<:Basis
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = rho0.data
    state = copy(rho0)
    dstate = copy(rho0)
    integrate(tspan_, df, x0, state, dstate, fout; kwargs...)
end


# Time derivative functions
#   * dmaster_h
#   * dmaster_nh
#   * dmaster_h_dynamic -> callback(t, rho) -> dmaster_h
#   * dmaster_nh_dynamic -> callback(t, rho) -> dmaster_nh
# dmaster_h and dmaster_nh provide specialized implementations depending on
# the type of the given decay rate object which can either be nothing, a vector
# or a matrix.

function dmaster_h(rho::T1, H::AbstractOperator{B,B},
                    rates::Nothing, J::Vector{T2}, Jdagger::Vector{T2},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B}}
    mul!(drho, H, rho, -1.0im, 0.0im)
    mul!(drho, rho, H, 1.0im, 1.0)
    for i=1:length(J)
        mul!(tmp, J[i], rho)
        mul!(drho, tmp, Jdagger[i], 1.0, 1.0)

        mul!(drho, Jdagger[i], tmp, -0.5, 1.0)

        mul!(tmp, rho, Jdagger[i])
        mul!(drho, tmp, J[i], -0.5, 1.0)
    end
    return drho
end

function dmaster_h(rho::T1, H::AbstractOperator{B,B},
                    rates::Vector{Float64}, J::Vector{T2}, Jdagger::Vector{T2},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B}}
    mul!(drho, H, rho, -1.0im, 0.0im)
    mul!(drho, rho, H, 1.0im, 1.0)
    for i=1:length(J)
        mul!(tmp, J[i], rho, rates[i], 0.0)
        mul!(drho, tmp, Jdagger[i], 1.0, 1.0)

        mul!(drho, Jdagger[i], tmp, -0.5, 1.0)

        mul!(tmp, rho, Jdagger[i], rates[i], 0.0)
        mul!(drho, tmp, J[i], -0.5, 1.0)
    end
    return drho
end

function dmaster_h(rho::T1, H::AbstractOperator{B,B},
                    rates::Matrix{Float64}, J::Vector{T2}, Jdagger::Vector{T2},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B}}
    mul!(drho, H, rho, -1.0im, 0.0im)
    mul!(drho, rho, H, 1.0im, 1.0)
    for j=1:length(J), i=1:length(J)
        mul!(tmp, J[i], rho, rates[i,j], 0.0)
        mul!(drho, tmp, Jdagger[j], 1.0, 1.0)

        mul!(drho, Jdagger[j], tmp, -0.5, 1.0)

        mul!(tmp, rho, Jdagger[j], rates[i,j], 0.0)
        mul!(drho, tmp, J[i], -0.5, 1.0)
    end
    return drho
end

function dmaster_nh(rho::T1, Hnh::T2, Hnh_dagger::T2,
                    rates::Nothing, J::Vector{T3}, Jdagger::Vector{T3},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B},T3<:AbstractOperator{B,B}}
    mul!(drho, Hnh, rho, -1.0im, 0.0)
    mul!(drho, rho, Hnh_dagger, 1.0im, 1.0)
    for i=1:length(J)
        mul!(tmp, J[i], rho)
        mul!(drho, tmp, Jdagger[i], 1.0, 1.0)
    end
    return drho
end

function dmaster_nh(rho::T1, Hnh::T2, Hnh_dagger::T2,
                    rates::Vector{Float64}, J::Vector{T3}, Jdagger::Vector{T3},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B},T3<:AbstractOperator{B,B}}
    mul!(drho, Hnh, rho, -1.0im, 0.0)
    mul!(drho, rho, Hnh_dagger, 1.0im, 1.0)
    for i=1:length(J)
        mul!(tmp, J[i], rho)
        mul!(drho, tmp, Jdagger[i], rates[i], 1.0)
    end
    return drho
end

function dmaster_nh(rho::T1, Hnh::T2, Hnh_dagger::T2,
                    rates::Matrix{Float64}, J::Vector{T3}, Jdagger::Vector{T3},
                    drho::T1, tmp::T1) where {B<:Basis,T1<:Operator{B,B},T2<:AbstractOperator{B,B},T3<:AbstractOperator{B,B}}
    mul!(drho, Hnh, rho, -1.0im, 0.0)
    mul!(drho, rho, Hnh_dagger, 1.0im, 1.0)
    for j=1:length(J), i=1:length(J)
        mul!(tmp, J[i], rho)
        mul!(drho, tmp, Jdagger[j], rates[i,j], 1.0)
    end
    return drho
end

function dmaster_h_dynamic(t::Float64, rho::T, f::Function,
                    rates::DecayRates,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    result = f(t, rho)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_master(rho, H, J, Jdagger, rates_)
    dmaster_h(rho, H, rates_, J, Jdagger, drho, tmp)
end

function dmaster_nh_dynamic(t::Float64, rho::T, f::Function,
                    rates::DecayRates,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    result = f(t, rho)
    QO_CHECKS[] && @assert 4 <= length(result) <= 5
    if length(result) == 4
        Hnh, Hnh_dagger, J, Jdagger = result
        rates_ = rates
    else
        Hnh, Hnh_dagger, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_master(rho, Hnh, J, Jdagger, rates_)
    dmaster_nh(rho, Hnh, Hnh_dagger, rates_, J, Jdagger, drho, tmp)
end


function check_master(rho0::Operator{B,B}, H::AbstractOperator{B,B}, J::Vector{T}, Jdagger::Vector{T}, rates::DecayRates) where {B<:Basis,T<:AbstractOperator{B,B}}
    isreducible = true # test if all operators are sparse or dense
    if !isa(H, Operator)
        isreducible = false
    end
    for j=J
        if !isa(j, Operator)
            isreducible = false
        end
    end
    for j=Jdagger
        if !isa(j, Operator)
            isreducible = false
        end
    end
    @assert length(J)==length(Jdagger)
    if typeof(rates) == Matrix{Float64}
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif typeof(rates) == Vector{Float64}
        @assert length(rates) == length(J)
    end
    isreducible
end

end #module
