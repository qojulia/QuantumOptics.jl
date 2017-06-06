module timeevolution_master

export master, master_nh, master_h, master_dynamic, master_nh_dynamic

import ..integrate, ..recast!
using ...bases
using ...states
using ...operators
using ...operators_dense
using ...operators_sparse

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Void}

"""
    timeevolution.master_h(tspan, rho0, H, J; <keyword arguments>)

Integrate the master equation with dmaster_h as derivative function.

Further information can be found at [`master`](@ref).
"""
function master_h(tspan, rho0::DenseOperator, H::Operator, J::Vector;
                rates::DecayRates=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    check_master(rho0, H, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
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
function master_nh(tspan, rho0::DenseOperator, Hnh::Operator, J::Vector;
                rates::DecayRates=nothing,
                Hnhdagger::Operator=dagger(Hnh),
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    check_master(rho0, Hnh, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
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
function master(tspan, rho0::DenseOperator, H::Operator, J::Vector;
                rates::DecayRates=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    isreducible = check_master(rho0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(rho0)
        dmaster_h_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
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
        dmaster_nh_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
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
function master_nh_dynamic(tspan, rho0::DenseOperator, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tmp = copy(rho0)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_nh_dynamic(t, rho, f, rates, drho, tmp)
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
function master_dynamic(tspan, rho0::DenseOperator, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Void}=nothing,
                kwargs...)
    tmp = copy(rho0)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_h_dynamic(t, rho, f, rates, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end


# Automatically convert Ket states to density operators
master(tspan, psi0::Ket, H::Operator, J::Vector; kwargs...) = master(tspan, dm(psi0), H, J; kwargs...)
master_h(tspan, psi0::Ket, H::Operator, J::Vector; kwargs...) = master_h(tspan, dm(psi0), H, J; kwargs...)
master_nh(tspan, psi0::Ket, Hnh::Operator, J::Vector; kwargs...) = master_nh(tspan, dm(psi0), Hnh, J; kwargs...)
master_dynamic(tspan, psi0::Ket, f::Function; kwargs...) = master_dynamic(tspan, dm(psi0), f; kwargs...)
master_nh_dynamic(tspan, psi0::Ket, f::Function; kwargs...) = master_nh_dynamic(tspan, dm(psi0), f; kwargs...)


# Recasting needed for the ODE solver is just providing the underlying data
function recast!(x::Vector{Complex128}, rho::DenseOperator)
    rho.data = reshape(x, size(rho.data))
end
recast!(rho::DenseOperator, x::Vector{Complex128}) = nothing

function integrate_master(tspan, df::Function, rho0::DenseOperator,
                        fout::Union{Void, Function}; kwargs...)
    tspan_ = convert(Vector{Float64}, tspan)
    x0 = reshape(rho0.data, length(rho0))
    state = DenseOperator(rho0.basis_l, rho0.basis_r, rho0.data)
    dstate = DenseOperator(rho0.basis_l, rho0.basis_r, rho0.data)
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

function dmaster_h(rho::DenseOperator, H::Operator,
                    rates::Void, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, H, rho, 0, drho)
    operators.gemm!(1im, rho, H, 1, drho)
    for i=1:length(J)
        operators.gemm!(1, J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[i], 1, drho)

        operators.gemm!(-0.5, Jdagger[i], tmp, 1, drho)

        operators.gemm!(1., rho, Jdagger[i], 0, tmp)
        operators.gemm!(-0.5, tmp, J[i], 1, drho)
    end
    return drho
end

function dmaster_h(rho::DenseOperator, H::Operator,
                    rates::Vector{Float64}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, H, rho, 0, drho)
    operators.gemm!(1im, rho, H, 1, drho)
    for i=1:length(J)
        operators.gemm!(rates[i], J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[i], 1, drho)

        operators.gemm!(-0.5, Jdagger[i], tmp, 1, drho)

        operators.gemm!(rates[i], rho, Jdagger[i], 0, tmp)
        operators.gemm!(-0.5, tmp, J[i], 1, drho)
    end
    return drho
end

function dmaster_h(rho::DenseOperator, H::Operator,
                    rates::Matrix{Float64}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, H, rho, 0, drho)
    operators.gemm!(1im, rho, H, 1, drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(rates[i,j], J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[j], 1, drho)

        operators.gemm!(-0.5, Jdagger[j], tmp, 1, drho)

        operators.gemm!(rates[i,j], rho, Jdagger[j], 0, tmp)
        operators.gemm!(-0.5, tmp, J[i], 1, drho)
    end
    return drho
end

function dmaster_nh(rho::DenseOperator, Hnh::Operator, Hnh_dagger::Operator,
                    rates::Void, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, Hnh, rho, 0, drho)
    operators.gemm!(1im, rho, Hnh_dagger, 1, drho)
    for i=1:length(J)
        operators.gemm!(1, J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    end
    return drho
end

function dmaster_nh(rho::DenseOperator, Hnh::Operator, Hnh_dagger::Operator,
                    rates::Vector{Float64}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, Hnh, rho, 0, drho)
    operators.gemm!(1im, rho, Hnh_dagger, 1, drho)
    for i=1:length(J)
        operators.gemm!(rates[i], J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[i], 1, drho)
    end
    return drho
end

function dmaster_nh(rho::DenseOperator, Hnh::Operator, Hnh_dagger::Operator,
                    rates::Matrix{Float64}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(-1im, Hnh, rho, 0, drho)
    operators.gemm!(1im, rho, Hnh_dagger, 1, drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(rates[i,j], J[i], rho, 0, tmp)
        operators.gemm!(1, tmp, Jdagger[j], 1, drho)
    end
    return drho
end

function dmaster_h_dynamic(t::Float64, rho::DenseOperator, f::Function,
                    rates::DecayRates,
                    drho::DenseOperator, tmp::DenseOperator)
    result = f(t, rho)
    @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    check_master(rho, H, J, Jdagger, rates_)
    dmaster_h(rho, H, rates_, J, Jdagger, drho, tmp)
end

function dmaster_nh_dynamic(t::Float64, rho::DenseOperator, f::Function,
                    rates::DecayRates,
                    drho::DenseOperator, tmp::DenseOperator)
    result = f(t, rho)
    @assert 4 <= length(result) <= 5
    if length(result) == 4
        Hnh, Hnh_dagger, J, Jdagger = result
        rates_ = rates
    else
        Hnh, Hnh_dagger, J, Jdagger, rates_ = result
    end
    check_master(rho, Hnh, J, Jdagger, rates_)
    dmaster_nh(rho, Hnh, Hnh_dagger, rates_, J, Jdagger, drho, tmp)
end


function check_master(rho0::DenseOperator, H::Operator, J::Vector, Jdagger::Vector, rates::DecayRates)
    isreducible = true # test if all operators are sparse or dense
    check_samebases(rho0, H)
    if !(isa(H, DenseOperator) || isa(H, SparseOperator))
        isreducible = false
    end
    for j=J
        @assert isa(j, Operator)
        if !(isa(j, DenseOperator) || isa(j, SparseOperator))
            isreducible = false
        end
        check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert isa(j, Operator)
        if !(isa(j, DenseOperator) || isa(j, SparseOperator))
            isreducible = false
        end
        check_samebases(rho0, j)
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
