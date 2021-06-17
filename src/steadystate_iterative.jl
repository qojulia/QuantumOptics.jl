using ..timeevolution: nh_hamiltonian, dmaster_h, dmaster_nh, check_master

"""
    iterative!(rho0, H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of master equation defined by a
Hamiltonian and a set of jump operators by solving `L rho = 0` via an iterative
method provided as argument.

# Arguments
* `rho0`: Initial density matrix. Note that this gets mutated in-place.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators.
* `method!`: The iterative method to be used. Defaults to `IterativeSolvers.bicgstabl!`.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates) for
    the jump operators. If nothing is specified all rates are assumed to be 1.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
    operators. If they are not given they are calculated automatically.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.
See also: [`iterative`](@ref)

Credit for this implementation goes to Z. Denis and F. Vicentini.
See also https://github.com/Z-Denis/SteadyState.jl
"""
function iterative!(rho0::Operator, H::AbstractOperator, J,
                    method! = IterativeSolvers.bicgstabl!, args...;
                    rates=nothing, Jdagger=dagger.(J), kwargs...)

    # Solution x must satisfy L*x = y with y[end] = tr(x) = 1 and y[j≠end] = 0.
    M = length(rho0.basis_l)
    x0 = zeros(eltype(rho0), M^2+1)
    x0[1:end-1] .= reshape(rho0.data, M^2)

    y = zeros(eltype(rho0), M^2+1)
    y[end] = one(eltype(rho0))

    # Define the linear map lm: rho ↦ L(rho)
    lm = _linmap_liouvillian(rho0,H,J,Jdagger,rates)

    log = get(kwargs,:log,false)

    # Solve the linear system with the iterative solver, then devectorize rho
    if !log
        rho0.data .= @views reshape(method!(x0,lm,y,args...;kwargs...)[1:end-1],(M,M))
        return rho0
    else
        R, history = method!(x0,lm,y,args...;kwargs...)
        rho0.data .= @views reshape(R[1:end-1],(M,M))
        return rho0, history
    end
end

"""
    iterative(H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of master equation defined by a
Hamiltonian and a set of jump operators by solving `L rho = 0` via an iterative
method provided as argument.

# Arguments
* `rho0`: Initial density matrix. Note that this gets mutated in-place.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators.
* `method!`: The iterative method to be used. Defaults to `IterativeSolvers.bicgstabl!`
    or `IterativeSolvers.idrs!` depending on arguments' types.
* `rates=nothing`: Vector or matrix specifying the coefficients (decay rates) for
    the jump operators. If nothing is specified all rates are assumed to be 1.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
    operators. If they are not given they are calculated automatically.
* `rho0=nothing`: Initial density operator.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.
See also: [`iterative!`](@ref)

Credit for this implementation goes to Z. Denis and F. Vicentini.
See also https://github.com/Z-Denis/SteadyState.jl
"""
function iterative(H::AbstractOperator, args...;
                rho0=nothing, kwargs...)
    if rho0 === nothing
        rho = DenseOperator(H.basis_l, H.basis_r)
        rho.data[1,1] = 1
    else
        rho = deepcopy(rho0)
    end
    return iterative!(rho, H, args...; kwargs...)
end


function _linmap_liouvillian(rho,H,J,Jdagger,rates)
    bl = rho.basis_l
    br = rho.basis_r
    M = length(bl)

    # Cache stuff
    drho = copy(rho)
    # rho = copy(rho)
    Jrho_cache = copy(rho)

    # Check reducibility
    isreducible = check_master(rho,H,J,Jdagger,rates)
    if isreducible
        Hnh = nh_hamiltonian(H,J,Jdagger,rates)
        Hnhdagger = dagger(Hnh)
        dmaster_ = (drho,rho) -> dmaster_nh(rho,Hnh,Hnhdagger,rates,J,Jdagger,drho,Jrho_cache)
    else
        dmaster_ = (drho,rho) -> dmaster_h(rho,H,rates,J,Jdagger,drho,Jrho_cache)
    end

    # Linear mapping
    function f!(y,x)
        # Reshape
        drho.data .= @views reshape(y[1:end-1], M, M)
        rho.data .= @views reshape(x[1:end-1], M, M)
        # Apply function
        dmaster_(drho,rho)
        # Recast data
        @views y[1:end-1] .= reshape(drho.data, M^2)
        y[end] = tr(rho)
        return y
    end

    return LinearMaps.LinearMap{eltype(rho)}(f!,M^2+1;ismutating=true,issymmetric=false,ishermitian=false,isposdef=false)
end
