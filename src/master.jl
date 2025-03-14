"""
    timeevolution.master_h(tspan, rho0, H, J; <keyword arguments>)

Integrate the master equation with dmaster_h as derivative function.

Further information can be found at [`master`](@ref).
"""
function master_h(tspan, rho0::Operator, H::AbstractOperator, J;
                rates=nothing,
                Jdagger=dagger.(J),
                fout=nothing,
                kwargs...)
    _check_const(H)
    _check_const.(J)
    _check_const.(Jdagger)
    check_master(rho0, H, J, Jdagger, rates)
    tspan, rho0 = _promote_time_and_state(rho0, H, J, tspan)
    tmp = copy(rho0)
    dmaster_(t, rho, drho) = dmaster_h!(drho, H, J, Jdagger, rates, rho, tmp)
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
function master_nh(tspan, rho0::Operator, Hnh::AbstractOperator, J;
                rates=nothing,
                Hnhdagger::AbstractOperator=dagger(Hnh),
                Jdagger=dagger.(J),
                fout=nothing,
                kwargs...)
    _check_const(Hnh)
    _check_const(Hnhdagger)
    _check_const.(J)
    _check_const.(Jdagger)
    check_master(rho0, Hnh, J, Jdagger, rates)
    tspan, rho0 = _promote_time_and_state(rho0, Hnh, J, tspan)
    tmp = copy(rho0)
    dmaster_(t, rho, drho) = dmaster_nh!(drho, Hnh, Hnhdagger, J, Jdagger, rates, rho, tmp)
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
function master(tspan, rho0::Operator, H::AbstractOperator, J;
                rates=nothing,
                Jdagger=dagger.(J),
                fout=nothing,
                kwargs...)
    _check_const(H)
    _check_const.(J)
    _check_const.(Jdagger)
    tspan, rho0 = _promote_time_and_state(rho0, H, J, tspan)
    isreducible = check_master(rho0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(rho0)
        dmaster_h_(t, rho, drho) = dmaster_h!(drho, H, J, Jdagger, rates, rho, tmp)
        return integrate_master(tspan, dmaster_h_, rho0, fout; kwargs...)
    else
        Hnh = nh_hamiltonian(H,J,Jdagger,rates)
        Hnhdagger = dagger(Hnh)
        tmp = copy(rho0)
        dmaster_nh_(t, rho, drho) = dmaster_nh!(drho, Hnh, Hnhdagger, J, Jdagger, rates, rho, tmp)
        return integrate_master(tspan, dmaster_nh_, rho0, fout; kwargs...)
    end
end

"""
    timeevolution.master(tspan, rho0, L; <keyword arguments>)

Time-evolution according to a master equation with a Liouvillian superoperator `L`.
# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `L`: Superoperator defining the right-hand-side of the master equation.
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.

See also: [`master_dynamic`](@ref)
"""
function master(tspan, rho0::Operator, L::SuperOperator; fout=nothing, kwargs...)
    # Rewrite rho as Ket and L as Operator
    dim = length(rho0.basis_l)*length(rho0.basis_r)
    b = GenericBasis(dim)
    rho_ = Ket(b,reshape(rho0.data, dim))
    L_ = Operator(b,b,L.data)
    tspan, rho_ = _promote_time_and_state(rho_, L_, tspan)
    dmaster_(t,rho,drho) = dmaster_liouville!(drho,L_,rho)

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
    timeevolution.master_nh_dynamic(tspan, rho0, f; <keyword arguments>)

Time-evolution according to a master equation with a dynamic non-hermitian Hamiltonian and J.

In this case the given Hamiltonian is assumed to be the non-hermitian version.
```math
H_{nh} = H - \\frac{i}{2} \\sum_k J^†_k J_k
```
The given function can either be of the form `f(t, rho) -> (Hnh, Hnhdagger, J, Jdagger)`
or `f(t, rho) -> (Hnh, Hnhdagger, J, Jdagger, rates)` For further information look
at [`master_dynamic`](@ref).

    timeevolution.master_nh_dynamic(tspan, rho0, Hnh::AbstractTimeDependentOperator, J; <keyword arguments>)

This version takes the non-hermitian Hamiltonian `Hnh` and jump operators `J` as time-dependent operators.
The jump operators may be `<: AbstractTimeDependentOperator` or other types
of operator.
"""
function master_nh_dynamic(tspan, rho0::Operator, f;
                rates=nothing,
                fout=nothing,
                kwargs...)
    tmp = copy(rho0)
    dmaster_(t, rho, drho) = dmaster_nh_dynamic!(drho, f, rates, rho, tmp, t)
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end

function master_nh_dynamic(tspan, rho0::Operator, Hnh::AbstractTimeDependentOperator, J;
    kwargs...)
    f = master_nh_dynamic_function(Hnh, J)
    master_nh_dynamic(tspan, rho0, f; kwargs...)
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

    timeevolution.master_dynamic(tspan, rho0, H::AbstractTimeDependentOperator, J; <keyword arguments>)

This version takes the Hamiltonian `H` and jump operators `J` as time-dependent operators.
The jump operators may be `<: AbstractTimeDependentOperator` or other types
of operator.
"""
function master_dynamic(tspan, rho0::Operator, f;
                rates=nothing,
                fout=nothing,
                kwargs...)
    tmp = copy(rho0)
    dmaster_ = let f = f, tmp = tmp
        dmaster_(t, rho, drho) = dmaster_h_dynamic!(drho, f, rates, rho, tmp, t)
    end
    integrate_master(tspan, dmaster_, rho0, fout; kwargs...)
end

function master_dynamic(tspan, rho0::Operator, H::AbstractTimeDependentOperator, J;
    kwargs...)
    f = master_h_dynamic_function(H, J)
    master_dynamic(tspan, rho0, f; kwargs...)
end

# Automatically convert Ket states to density operators
for f ∈ [:master,:master_h,:master_nh,:master_dynamic,:master_nh_dynamic]
    @eval $f(tspan,psi0::Ket,args...;kwargs...) = $f(tspan,dm(psi0),args...;kwargs...)
end

# Non-hermitian Hamiltonian
function nh_hamiltonian(H,J,Jdagger,::Nothing)
    Hnh = copy(H)
    for i=1:length(J)
        Hnh -= complex(float(eltype(H)))(0.5im)*Jdagger[i]*J[i]
    end
    return Hnh
end
function nh_hamiltonian(H,J,Jdagger,rates::AbstractVector)
    Hnh = copy(H)
    for i=1:length(J)
        Hnh -= complex(float(eltype(H)))(0.5im*rates[i])*Jdagger[i]*J[i]
    end
    return Hnh
end
function nh_hamiltonian(H,J,Jdagger,rates::AbstractMatrix)
    Hnh = copy(H)
    for i=1:length(J), j=1:length(J)
        Hnh -= complex(float(eltype(H)))(0.5im*rates[i,j])*Jdagger[i]*J[j]
    end
    return Hnh
end

# Recasting needed for the ODE solver is just providing the underlying data
function recast!(rho::Operator{B,B,T},x::T) where {B,T}
    rho.data = x
end
recast!(x::T,rho::Operator{B,B,T}) where {B,T} = nothing

function integrate_master(tspan, df, rho0, fout; kwargs...)
    x0 = rho0.data
    state = deepcopy(rho0)
    dstate = deepcopy(rho0)
    integrate(tspan, df, x0, state, dstate, fout; kwargs...)
end


# Time derivative functions
#   * dmaster_h
#   * dmaster_nh
#   * dmaster_h_dynamic -> callback(t, rho) -> dmaster_h
#   * dmaster_nh_dynamic -> callback(t, rho) -> dmaster_nh
# dmaster_h and dmaster_nh provide specialized implementations depending on
# the type of the given decay rate object which can either be nothing, a vector
# or a matrix.

"""
    dmaster_h!(drho, H, J, Jdagger, rates, rho, drho_cache)

Update `drho` according to a master equation given in standard Lindblad form.
A cached copy `drho_cache` of `drho` is used as a temporary saving step.

See also: [`master`](@ref), [`dmaster_nh!`](@ref), [`dmaster_h_dynamic!`](@ref),
    [`dmaster_nh_dynamic!`](@ref), [`dmaster_liouville!`](@ref)
"""
function dmaster_h!(drho, H, J, Jdagger, rates::Nothing, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho)
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[i],true,false)
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_h!(drho, H, J, Jdagger, rates::AbstractVector, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho,eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[i],eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_h!(drho, H, J, Jdagger, rates::AbstractMatrix, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,eltype(rho)(im),one(eltype(rho)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho,eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[j],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[j],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[j],eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

"""
    dmaster_nh!(drho, Hnh, Hnh_dagger, J, Jdagger, rates, rho, drho_cache)

Updates `drho` according to a master equation given in standard Lindblad form.
The part of the Liuovillian which can be written as a commutator should be
contained in `Hnh` and `Hnh_dagger`. This allows to skip a number of matrix
multiplications making it slightly faster than [`dmaster_h!`](@ref).

See also: [`master`](@ref), [`dmaster_h!`](@ref), [`dmaster_h_dynamic!`](@ref),
    [`dmaster_nh_dynamic!`](@ref), [`dmaster_liouville!`](@ref)
"""
function dmaster_nh!(drho, Hnh, Hnh_dagger, J, Jdagger, rates::Nothing, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho)
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[i],true,true)
    end
    return drho
end

function dmaster_nh!(drho, Hnh, Hnh_dagger, J, Jdagger, rates::AbstractVector, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho,eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[i],true,true)
    end
    return drho
end

function dmaster_nh!(drho, Hnh, Hnh_dagger, J, Jdagger, rates::AbstractMatrix, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,Hnh,rho,-eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,Hnh_dagger,eltype(rho)(im),one(eltype(rho)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,J[i],rho,eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,Jdagger[j],true,true)
    end
    return drho
end

"""
    dmaster_liouville!(drho,L,rho)

Update `drho` according to a master equation as `L*rho`, where `L` is an arbitrary
(super-)operator.

See also: [`master`](@ref), [`dmaster_h!`](@ref), [`dmaster_nh!`](@ref),
    [`dmaster_h_dynamic!`](@ref), [`dmaster_nh_dynamic!`](@ref)
"""
function dmaster_liouville!(drho,L,rho)
    mul!(drho,L,rho)
    return drho
end

"""
    dmaster_h_dynamic!(drho, f, rates, rho, drho_cache, t)

Computes the Hamiltonian and jump operators as `H,J,Jdagger=f(t,rho)` and
update `drho` according to a master equation. Optionally, rates can also be
returned from `f`.

See also: [`master_dynamic`](@ref), [`dmaster_h!`](@ref), [`dmaster_nh!`](@ref),
    [`dmaster_nh_dynamic!`](@ref)
"""
function dmaster_h_dynamic!(drho, f::F, rates, rho, drho_cache, t) where {F}
    result = f(t, rho)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_master(rho, H, J, Jdagger, rates_)
    dmaster_h!(drho, H, J, Jdagger, rates_, rho, drho_cache)
end

"""
    dmaster_nh_dynamic!(drho, f, rates, rho, drho_cache, t)

Computes the non-hermitian Hamiltonian and jump operators as
`Hnh,Hnh_dagger,J,Jdagger=f(t,rho)` and update `drho` according to a master
equation. Optionally, rates can also be returned from `f`.

See also: [`master_dynamic`](@ref), [`dmaster_h!`](@ref), [`dmaster_nh!`](@ref),
    [`dmaster_h_dynamic!`](@ref)
"""
function dmaster_nh_dynamic!(drho, f::F, rates, rho, drho_cache, t) where {F}
    result = f(t, rho)
    QO_CHECKS[] && @assert 4 <= length(result) <= 5
    if length(result) == 4
        Hnh, Hnh_dagger, J, Jdagger = result
        rates_ = rates
    else
        Hnh, Hnh_dagger, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_master(rho, Hnh, J, Jdagger, rates_)
    dmaster_nh!(drho, Hnh, Hnh_dagger, J, Jdagger, rates_, rho, drho_cache)
end


function check_master(rho0, H, J, Jdagger, rates)
    isreducible = true # test if all operators are sparse or dense
    if !(isa(H, DenseOpType) || isa(H, SparseOpType))
        isreducible = false
    end
    for j=J
        @assert isa(j, AbstractOperator)
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
        check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator)
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
        check_samebases(rho0, j)
    end
    @assert length(J)==length(Jdagger)
    if isa(rates,AbstractMatrix)
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif isa(rates,AbstractVector)
        @assert length(rates) == length(J)
    end
    isreducible
end
