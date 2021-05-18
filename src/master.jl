const DecayRates = Union{Vector, Matrix, Nothing}

"""
    timeevolution.master_h(tspan, rho0, H, J; <keyword arguments>)

Integrate the master equation with dmaster_h as derivative function.

Further information can be found at [`master`](@ref).
"""
function master_h(tspan, rho0::T, H::AbstractOperator{B,B}, J::Vector;
                rates::DecayRates=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}
    check_master(rho0, H, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::T, drho::T) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
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
function master_nh(tspan, rho0::T, Hnh::AbstractOperator{B,B}, J::Vector;
                rates::DecayRates=nothing,
                Hnhdagger::AbstractOperator=dagger(Hnh),
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}
    check_master(rho0, Hnh, J, Jdagger, rates)
    tmp = copy(rho0)
    dmaster_(t, rho::T, drho::T) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
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
function master(tspan, rho0::T, H::AbstractOperator{B,B}, J::Vector;
                rates::DecayRates=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}
    isreducible = check_master(rho0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(rho0)
        dmaster_h_(t, rho::T, drho::T) = dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        return integrate_master(tspan, dmaster_h_, rho0, fout; kwargs...)
    else
        Hnh = copy(H)
        if isa(rates, Matrix)
            for i=1:length(J), j=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im*rates[i,j])*Jdagger[i]*J[j]
            end
        elseif isa(rates, Vector)
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im*rates[i])*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im)*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)
        tmp = copy(rho0)
        dmaster_nh_(t, rho::T, drho::T) = dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        return integrate_master(tspan, dmaster_nh_, rho0, fout; kwargs...)
    end
end

function master(tspan, rho0::Operator, L::SuperOperator; fout=nothing, kwargs...)
    # Rewrite rho as Ket and L as Operator
    dim = length(rho0.basis_l)*length(rho0.basis_r)
    b = GenericBasis(dim)
    rho_ = Ket(b,reshape(rho0.data, dim))
    L_ = Operator(b,b,L.data)
    dmaster_(t,rho,drho) = dmaster_liouville(rho,drho,L_)

    # Rewrite into density matrix when saving
    tmp = copy(rho0)
    if fout===nothing
        fout_ = function(t,rho)
            tmp.data[:] = rho.data
            return copy(tmp)
        end
    else
        fout_ = function(t,rho)
            tmp.data[:] = rho.data
            return fout(t,tmp)
        end
    end

    # Solve
    return integrate_master(tspan, dmaster_, rho_, fout_; kwargs...)
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
function master_nh_dynamic(tspan, rho0::T, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}
    tmp = copy(rho0)
    dmaster_(t, rho::T, drho::T) = dmaster_nh_dynamic(t, rho, f, rates, drho, tmp)
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
function master_dynamic(tspan, rho0::T, f::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}
    tmp = copy(rho0)
    dmaster_(t, rho::T, drho::T) = dmaster_h_dynamic(t, rho, f, rates, drho, tmp)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end


# Automatically convert Ket states to density operators
master(tspan, psi0::Ket{B}, args...; kwargs...) where B<:Basis = master(tspan, dm(psi0), args...; kwargs...)
master_h(tspan, psi0::Ket{B}, H::AbstractOperator{B,B}, J::Vector; kwargs...) where B<:Basis = master_h(tspan, dm(psi0), H, J; kwargs...)
master_nh(tspan, psi0::Ket{B}, Hnh::AbstractOperator{B,B}, J::Vector; kwargs...) where B<:Basis = master_nh(tspan, dm(psi0), Hnh, J; kwargs...)
master_dynamic(tspan, psi0::Ket{B}, f::Function; kwargs...) where B<:Basis = master_dynamic(tspan, dm(psi0), f; kwargs...)
master_nh_dynamic(tspan, psi0::Ket{B}, f::Function; kwargs...) where B<:Basis = master_nh_dynamic(tspan, dm(psi0), f; kwargs...)


# Recasting needed for the ODE solver is just providing the underlying data
function recast!(x::T, rho::Operator{B,B,T}) where {B<:Basis,T}
    rho.data = x
end
recast!(rho::Operator{B,B,T}, x::T) where {B<:Basis,T} = nothing

function integrate_master(tspan, df::Function, rho0,
                        fout::Union{Nothing, Function}; kwargs...)
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    x0 = rho0.data
    state = deepcopy(rho0)
    dstate = deepcopy(rho0)
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

function dmaster_h(rho::T, H::AbstractOperator{B,B},
                    rates::Nothing, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho)
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],tmp,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(tmp,rho,Jdagger[i],true,false)
        QuantumOpticsBase.mul!(drho,tmp,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_h(rho::T, H::AbstractOperator{B,B},
                    rates::Vector, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho,eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],tmp,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(tmp,rho,Jdagger[i],eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_h(rho::T, H::AbstractOperator{B,B},
                    rates::Matrix, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho,eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[j],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[j],tmp,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(tmp,rho,Jdagger[j],eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_nh(rho::T, Hnh::AbstractOperator{B,B}, Hnh_dagger::AbstractOperator{B,B},
                    rates::Nothing, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho)
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[i],true,true)
    end
    return drho
end

function dmaster_nh(rho::T, Hnh::AbstractOperator{B,B}, Hnh_dagger::AbstractOperator{B,B},
                    rates::Vector, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho,eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[i],true,true)
    end
    return drho
end

function dmaster_nh(rho::T, Hnh::AbstractOperator{B,B}, Hnh_dagger::AbstractOperator{B,B},
                    rates::Matrix, J::Vector, Jdagger::Vector,
                    drho::T, tmp::T) where {B<:Basis,T<:Operator{B,B}}
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(tmp,J[i],rho,eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,tmp,Jdagger[j],true,true)
    end
    return drho
end

function dmaster_liouville(rho,drho,L)
    mul!(drho,L,rho)
    return drho
end

function dmaster_h_dynamic(t, rho::T, f::Function,
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

function dmaster_nh_dynamic(t, rho::T, f::Function,
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


function check_master(rho0::Operator{B,B}, H::AbstractOperator{B,B}, J::Vector, Jdagger::Vector, rates::DecayRates) where B<:Basis
    # TODO: clean up type checks by dispatch; make type of J known
    isreducible = true # test if all operators are sparse or dense
    if !(isa(H, DenseOpType) || isa(H, SparseOpType))
        isreducible = false
    end
    for j=J
        @assert isa(j, AbstractOperator{B,B})
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
        check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator{B,B})
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
        check_samebases(rho0, j)
    end
    @assert length(J)==length(Jdagger)
    if isa(rates,Matrix)
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif isa(rates,Vector)
        @assert length(rates) == length(J)
    end
    isreducible
end

get_type(rho, H, rates::Nothing, J, Jdagger) = promote_type(eltype(rho),eltype(H),eltype.(J)...,eltype.(Jdagger)...)
get_type(rho, H, rates::Union{Vector,Matrix}, J, Jdagger) = promote_type(eltype(rho),eltype(H),eltype(rates),eltype.(J)...,eltype.(Jdagger)...)
get_type(rho, Hnh, Hnhdagger, rates::Nothing, J, Jdagger) = promote_type(eltype(rho),eltype(Hnh),eltype(Hnhdagger), eltype.(J)...,eltype.(Jdagger)...)
get_type(rho, Hnh, Hnhdagger, rates::Union{Vector,Matrix}, J, Jdagger) = promote_type(eltype(rho),eltype(Hnh),eltype(Hnhdagger),eltype(rates),eltype.(J)...,eltype.(Jdagger)...)
