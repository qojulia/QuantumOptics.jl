import ...timeevolution: dmaster_h, dmaster_nh, dmaster_h_dynamic, check_master

const DecayRates = Union{Vector, Matrix, Nothing}

"""
    stochastic.master(tspan, rho0, H, J, C; <keyword arguments>)

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
* `C`: Vector containing the stochastic operators for a superoperator
        of the form `C[i]*rho + rho*Cdagger[i]`.
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
function master(tspan, rho0::T, H::AbstractOperator{B,B},
                J::Vector, C::Vector;
                rates::DecayRates=nothing,
                Jdagger::Vector=dagger.(J), Cdagger::Vector=dagger.(C),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}

    tmp = copy(rho0)

    n = length(C)

    dmaster_stoch(dx::DiffArray, t, rho::T, drho::T, n::Int) =
            dmaster_stochastic(dx, rho, C, Cdagger, drho, n)

    isreducible = check_master(rho0, H, J, Jdagger, rates) && check_master_stoch(rho0, C, Cdagger)
    if !isreducible
        dmaster_h_determ(t, rho::T, drho::T) =
            dmaster_h(rho, H, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_h_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    else
        Hnh = copy(H)
        if isa(rates, Matrix)
            for i=1:length(J), j=1:length(J)
                Hnh -= complex(eltype(H))(0.5im*rates[i,j])*Jdagger[i]*J[j]
            end
        elseif isa(rates, Vector)
            for i=1:length(J)
                Hnh -= complex(eltype(H))(0.5im*rates[i])*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= complex(eltype(H))(0.5im)*Jdagger[i]*J[i]
            end
        end
        Hnhdagger = dagger(Hnh)

        dmaster_nh_determ(t, rho::T, drho::T) =
            dmaster_nh(rho, Hnh, Hnhdagger, rates, J, Jdagger, drho, tmp)
        integrate_master_stoch(tspan, dmaster_nh_determ, dmaster_stoch, rho0, fout, n; kwargs...)
    end
end
master(tspan, psi0::Ket, args...; kwargs...) = master(tspan, dm(psi0), args...; kwargs...)

"""
    stochastic.master_dynamic(tspan, rho0, fdeterm, fstoch; <keyword arguments>)

Time-evolution according to a stochastic master equation with a
dynamic Hamiltonian and J.

# Arguments
* `tspan`: Vector specifying the points of time for which output should be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `fdeterm`: Function `f(t, rho) -> (H, J, Jdagger)` or
        `f(t, rho) -> (H, J, Jdagger, rates)` giving the deterministic
        part of the master equation.
* `fstoch`: Function `f(t, rho) -> (C, Cdagger)` giving the stochastic
        superoperator of the form `C[i]*rho + rho*Cdagger[i]`.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators. If nothing is specified all rates are assumed
        to be 1.
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
function master_dynamic(tspan, rho0::T, fdeterm::Function, fstoch::Function;
                rates::DecayRates=nothing,
                fout::Union{Function,Nothing}=nothing,
                noise_processes::Int=0,
                kwargs...) where {B<:Basis,T<:Operator{B,B}}

    tmp = copy(rho0)

    if noise_processes == 0
        fs_out = fstoch(0, rho0)
        n = length(fs_out[1])
    else
        n = noise_processes
    end

    dmaster_determ(t, rho::T, drho::T) = dmaster_h_dynamic(t, rho, fdeterm, rates, drho, tmp)
    dmaster_stoch(dx::DiffArray, t, rho::T, drho::T, n::Int) =
        dmaster_stoch_dynamic(dx, t, rho, fstoch, drho, n)
    integrate_master_stoch(tspan, dmaster_determ, dmaster_stoch, rho0, fout, n; kwargs...)
end
master_dynamic(tspan, psi0::Ket, args...; kwargs...) = master_dynamic(tspan, dm(psi0), args...; kwargs...)

# Derivative functions
function dmaster_stochastic(dx::Vector, rho::T,
            C::Vector, Cdagger::Vector, drho::T, ::Int) where {B<:Basis,T<:Operator{B,B}}
    recast!(dx, drho)
    QuantumOpticsBase.mul!(drho,C[1],rho)
    QuantumOpticsBase.mul!(drho,rho,Cdagger[1],true,true)
    drho.data .-= tr(drho)*rho.data
end
function dmaster_stochastic(dx::Matrix, rho::T,
            C::Vector, Cdagger::Vector, drho::T, n::Int) where {B<:Basis,T<:Operator{B,B}}
    for i=1:n
        dx_i = @view dx[:, i]
        recast!(dx_i, drho)
        QuantumOpticsBase.mul!(drho,C[i],rho)
        QuantumOpticsBase.mul!(drho,rho,Cdagger[i],true,true)
        drho.data .-= tr(drho)*rho.data
        recast!(drho, dx_i)
    end
end

function dmaster_stoch_dynamic(dx::DiffArray, t, rho::T,
            f::Function, drho::T, n::Int) where {B<:Basis,T<:Operator{B,B}}
    result = f(t, rho)
    QO_CHECKS[] && @assert 2 == length(result)
    C, Cdagger = result
    QO_CHECKS[] && check_master_stoch(rho, C, Cdagger)
    dmaster_stochastic(dx, rho, C, Cdagger, drho, n)
end

function integrate_master_stoch(tspan, df::Function, dg::Function,
                        rho0::Operator, fout::Union{Nothing, Function},
                        n::Int;
                        kwargs...)
    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    x0 = reshape(rho0.data, length(rho0))
    state = copy(rho0)
    dstate = copy(rho0)
    integrate_stoch(tspan_, df, dg, x0, state, dstate, fout, n; kwargs...)
end

function check_master_stoch(rho0::Operator{B,B}, C::Vector, Cdagger::Vector) where B<:Basis
    # TODO: replace type checks by dispatch; make types of C known
    @assert length(C) == length(Cdagger)
    isreducible = true
    for c=C
        @assert isa(c, AbstractOperator{B,B})
        if !isa(c, DataOperator)
            isreducible = false
        end
    end
    for c=Cdagger
        @assert isa(c, AbstractOperator{B,B})
        if !isa(c, DataOperator)
            isreducible = false
        end
    end
    isreducible
end


# TODO: Speed up by recasting to n-d arrays, remove vector methods
function recast!(x::Union{Vector, SubArray}, rho::Operator{B,B,T}) where {B<:Basis,T}
    rho.data = reshape(x, size(rho.data))
end
recast!(state::Operator{B,B}, x::SubArray) where B<:Basis = (x[:] = state.data)
recast!(state::Operator{B,B}, x::Vector) where B<:Basis = nothing
