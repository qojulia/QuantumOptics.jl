using Random, LinearAlgebra

import ..timeevolution: dschroedinger!

# TODO: Remove imports
import RecursiveArrayTools.copyat_or_push!

"""
    mcwf_h(tspan, rho0, Hnh, J; <keyword arguments>)

Calculate MCWF trajectory where the Hamiltonian is given in hermitian form.

For more information see: [`mcwf`](@ref)
"""
function mcwf_h(tspan, psi0::Ket, H::AbstractOperator, J;
        seed=rand(UInt), rates=nothing,
        fout=nothing, Jdagger=dagger.(J),
        tmp=copy(psi0),
        display_beforeevent=false, display_afterevent=false,
        kwargs...)
    check_mcwf(psi0, H, J, Jdagger, rates)
    f(t, psi, dpsi) = dmcwf_h!(dpsi, H, J, Jdagger, rates, psi, tmp)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
    integrate_mcwf(f, j, tspan, psi0, seed, fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

"""
    mcwf_nh(tspan, rho0, Hnh, J; <keyword arguments>)

Calculate MCWF trajectory where the Hamiltonian is given in non-hermitian form.

```math
H_{nh} = H - \\frac{i}{2} \\sum_k J^†_k J_k
```

For more information see: [`mcwf`](@ref)
"""
function mcwf_nh(tspan, psi0::Ket, Hnh::AbstractOperator, J;
        seed=rand(UInt), fout=nothing,
        display_beforeevent=false, display_afterevent=false,
        kwargs...)
    check_mcwf(psi0, Hnh, J, J, nothing)
    f(t, psi, dpsi) = dschroedinger!(dpsi, Hnh, psi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, nothing)
    integrate_mcwf(f, j, tspan, psi0, seed, fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

"""
    mcwf(tspan, psi0, H, J; <keyword arguments>)

Integrate the master equation using the MCWF method.

There are two implementations for integrating the non-hermitian
schroedinger equation:

* [`mcwf_h`](@ref): Usual formulation with Hamiltonian + jump operators
separately.
* [`mcwf_nh`](@ref): Variant with non-hermitian Hamiltonian.

The `mcwf` function takes a normal Hamiltonian, calculates the
non-hermitian Hamiltonian and then calls [`mcwf_nh`](@ref) which is
slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector.
* `H`: Arbitrary Operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
operator type.
* `seed=rand()`: Seed used for the random number generator.
* `rates=ones()`: Vector of decay rates.
* `fout`: If given, this function `fout(t, psi)` is called every time an
output should be displayed. ATTENTION: The state `psi` is neither
normalized nor permanent! It is still in use by the ode solve
and therefore must not be changed.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
operators. If they are not given they are calculated automatically.
* `display_beforeevent=false`: `fout` is called before every jump.
* `display_afterevent=false`: `fout` is called after every jump.
* `rng_state`: An optional `timeevolution.JumpRNGState``, providing the RNG and
        an initial jump threshold. If provided, `seed` is ignored.
* `display_jumps=false`: If set to true, an additional list of times and indices
is returned. These correspond to the times at which a jump occured and the index
of the jump operators with which the jump occured, respectively.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf(tspan, psi0::Ket, H::AbstractOperator, J;
        seed=rand(UInt), rates=nothing,
        fout=nothing, Jdagger=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        kwargs...)
    isreducible = check_mcwf(psi0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(psi0)
        dmcwf_h_(t, psi, dpsi) = dmcwf_h!(dpsi, H, J, Jdagger, rates, psi, tmp)
        j_h(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
        integrate_mcwf(dmcwf_h_, j_h, tspan, psi0, seed,
            fout;
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            kwargs...)
    else
        Hnh = copy(H)
        if isa(rates, Nothing)
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im)*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= complex(float(eltype(H)))(0.5im*rates[i])*Jdagger[i]*J[i]
            end
        end
        dmcwf_nh_(t, psi, dpsi) = dschroedinger!(dpsi, Hnh, psi)
        j_nh(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
        integrate_mcwf(dmcwf_nh_, j_nh, tspan, psi0, seed,
            fout;
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            kwargs...)
    end
end

"""
    mcwf_dynamic(tspan, psi0, f; <keyword arguments>)

Integrate the master equation using the MCWF method with dynamic
Hamiltonian and Jump operators.

The `mcwf` function takes a normal Hamiltonian, calculates the
non-hermitian Hamiltonian and then calls [`mcwf_nh`](@ref) which is
slightly faster.

# Arguments
* `tspan`: Vector specifying the points of time for which output should
be displayed.
* `psi0`: Initial state vector.
* `f`: Function `f(t, psi) -> (H, J, Jdagger)` or `f(t, psi) -> (H, J, Jdagger, rates)`
    that returns the time-dependent Hamiltonian and Jump operators.
* `seed=rand()`: Seed used for the random number generator.
* `rates=ones()`: Vector of decay rates.
* `fout`: If given, this function `fout(t, psi)` is called every time an
output should be displayed. ATTENTION: The state `psi` is neither
normalized nor permanent! It is still in use by the ode solve
and therefore must not be changed.
* `display_beforeevent=false`: `fout` is called before every jump.
* `display_afterevent=false`: `fout` is called after every jump.
* `rng_state`: An optional `timeevolution.JumpRNGState``, providing the RNG and
        an initial jump threshold. If provided, `seed` is ignored.
* `display_jumps=false`: If set to true, an additional list of times and indices
is returned. These correspond to the times at which a jump occured and the index
of the jump operators with which the jump occured, respectively.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf_dynamic(tspan, psi0::Ket, f;
    seed=rand(UInt), rates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...)
    tmp = copy(psi0)
    dmcwf_(t, psi, dpsi) = dmcwf_h_dynamic!(dpsi, f, rates, psi, tmp, t)
    j_(rng, t, psi, psi_new) = jump_dynamic(rng, t, psi, f, psi_new, rates)
    integrate_mcwf(dmcwf_, j_, tspan, psi0, seed,
        fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

"""
    mcwf_nh_dynamic(tspan, rho0, f; <keyword arguments>)

Calculate MCWF trajectory where the dynamic Hamiltonian is given in non-hermitian form.

For more information see: [`mcwf_dynamic`](@ref)
"""
function mcwf_nh_dynamic(tspan, psi0::Ket, f;
    seed=rand(UInt), rates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...)
    dmcwf_(t, psi, dpsi) = dmcwf_nh_dynamic!(dpsi, f, psi, t)
    j_(rng, t, psi, psi_new) = jump_dynamic(rng, t, psi, f, psi_new, rates)
    integrate_mcwf(dmcwf_, j_, tspan, psi0, seed,
        fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

"""
    dmcwf_h_dynamic!(dpsi, f, rates, psi, dpsi_cache, t)

Compute the Hamiltonian and jump operators as `H,J,Jdagger=f(t,psi)` and
update `dpsi` according to a non-Hermitian Schrödinger equation.

See also: [`mcwf_dynamic`](@ref), [`dmcwf_h!`](@ref), [`dmcwf_nh_dynamic`](@ref)
"""
function dmcwf_h_dynamic!(dpsi, f, rates, psi, dpsi_cache, t)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, rates_)
    dmcwf_h!(dpsi, H, J, Jdagger, rates_, psi, dpsi_cache)
end

"""
    dmcwf_nh_dynamic!(dpsi, f, psi, t)

Compute the non-Hermitian Hamiltonian and jump operators as `H,J,Jdagger=f(t,psi)`
and update `dpsi` according to a Schrödinger equation.

See also: [`mcwf_nh_dynamic`](@ref), [`dmcwf_nh!`](@ref), [`dschroedinger!`](@ref)
"""
function dmcwf_nh_dynamic!(dpsi, f, psi, t)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    H, J, Jdagger = result[1:3]
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, nothing)
    dschroedinger!(dpsi, H, psi)
end

function jump_dynamic(rng, t, psi, f, psi_new, rates)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    J = result[2]
    if length(result) == 3
        rates_ = rates
    else
        rates_ = result[4]
    end
    jump(rng, t, psi, J, psi_new, rates_)
end

"""
    integrate_mcwf(dmcwf, jumpfun, tspan, psi0, seed; fout, kwargs...)

Integrate a single Monte Carlo wave function trajectory.

# Arguments
* `dmcwf`: A function `f(t, psi, dpsi)` that calculates the time-derivative of
        `psi` at time `t` and stores the result in `dpsi`.
* `jumpfun`: A function `f(rng, t, psi, dpsi)` that uses the random number
        generator `rng` to determine if a jump is performed and stores the
        result in `dpsi`.
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `psi0`: Initial state vector.
* `seed`: Seed used for the random number generator.
* `fout`: If given, this function `fout(t, psi)` is called every time an
        output should be displayed. ATTENTION: The state `psi` is neither
        normalized nor permanent! It is still in use by the ode solver
        and therefore must not be changed.
* `rng_state`: An optional `timeevolution.JumpRNGState``, providing the RNG and
        an initial jump threshold. If provided, `seed` is ignored.
* `kwargs`: Further arguments are passed on to the ode solver.
"""
function integrate_mcwf(dmcwf, jumpfun, tspan,
                        psi0, seed, fout::Function;
                        display_beforeevent=false, display_afterevent=false,
                        display_jumps=false,
                        rng_state=nothing,
                        save_everystep=false, callback=nothing,
                        saveat=tspan,
                        alg=OrdinaryDiffEq.DP5(),
                        kwargs...)

    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)
    # Display before or after events
    function save_func!(affect!,integrator)
        affect!.saveiter += 1
        copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
        copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
            affect!.save_func(integrator.u, integrator.t, integrator),Val{false})
        return nothing
    end
    save_before! = display_beforeevent ? save_func! : (affect!,integrator)->nothing
    save_after! = display_afterevent ? save_func! : (affect!,integrator)->nothing

    # Display jump operator index and times
    jump_t = eltype(tspan_)[]
    jump_index = Int[]
    save_t_index = if display_jumps
        function(t,i)
            push!(jump_t,t)
            push!(jump_index,i)
            return nothing
        end
    else
        (t,i)->nothing
    end

    function fout_(x, t, integrator)
        recast!(state,x)
        fout(t, state)
    end

    state = copy(psi0)
    dstate = copy(psi0)
    out_type = pure_inference(fout, Tuple{eltype(tspan_),typeof(state)})
    out = DiffEqCallbacks.SavedValues(eltype(tspan_),out_type)
    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan_,
                                         save_everystep=save_everystep,
                                         save_start = false)

    cb = jump_callback(jumpfun, seed, scb, save_before!, save_after!, save_t_index, psi0, rng_state)
    full_cb = OrdinaryDiffEq.CallbackSet(callback,cb,scb)

    function df_(dx, x, p, t)
        recast!(state,x)
        recast!(dstate,dx)
        dmcwf(t, state, dstate)
        recast!(dx,dstate)
    end

    prob = OrdinaryDiffEq.ODEProblem{true}(df_, as_vector(psi0), (tspan_[1],tspan_[end]))

    sol = OrdinaryDiffEq.solve(
                prob,
                alg;
                reltol = 1.0e-6,
                abstol = 1.0e-8,
                save_everystep = false, save_start = false,
                save_end = false,
                callback=full_cb, kwargs...)

    if display_jumps
        return out.t, out.saveval, jump_t, jump_index
    else
        return out.t, out.saveval
    end
end

function integrate_mcwf(dmcwf, jumpfun, tspan,
                        psi0, seed, fout::Nothing;
                        kwargs...)
    function fout_(t, x)
        return normalize(x)
    end
    integrate_mcwf(dmcwf, jumpfun, tspan, psi0, seed, fout_; kwargs...)
end

"""
Jump RNG state.

Stores the RNG used to generate jump thresholds, as well as
the most recent threshold rolled. A jump is carried out when
the norm-squared of the evolving state drops below the threshold.

Can be passed to `mcwf()` and related functions as the `rng_state`
keyword argument to persist the state across calls.
"""
mutable struct JumpRNGState{T<:Number,R<:AbstractRNG}
    rng::R
    threshold::T
end
function JumpRNGState(::Type{T}, seed) where T
    rng = MersenneTwister(seed)
    threshold = rand(rng, T)
    JumpRNGState(rng, threshold)
end
roll!(s::JumpRNGState{T}) where T = (s.threshold = rand(s.rng, T))
threshold(s::JumpRNGState) = s.threshold

function jump_callback(jumpfun, seed, scb, save_before!,
                        save_after!, save_t_index, psi0, rng_state)

    tmp = copy(psi0)
    psi_tmp = copy(psi0)

    if rng_state === nothing
        rngstate = JumpRNGState(real(eltype(psi0)), seed)
    else
        rngstate = rng_state
    end
    djumpnorm(x, t, integrator) = norm(x)^2 - (1-threshold(rngstate))

    function dojump(integrator)
        x = integrator.u
        t = integrator.t

        affect! = scb.affect!
        save_before!(affect!,integrator)
        recast!(psi_tmp,x)
        i = jumpfun(rngstate.rng, t, psi_tmp, tmp)
        x .= tmp.data
        save_after!(affect!,integrator)
        save_t_index(t,i)

        roll!(rngstate)
        return nothing
    end

    return OrdinaryDiffEq.ContinuousCallback(djumpnorm,dojump,
            save_positions = (false,false))
end
as_vector(psi::StateVector) = psi.data

"""
    jump(rng, t, psi, J, psi_new)

Default jump function.

# Arguments
* `rng:` Random number generator
* `t`: Point of time where the jump is performed.
* `psi`: State vector before the jump.
* `J`: List of jump operators.
* `psi_new`: Result of jump.
"""
function jump(rng, t, psi, J, psi_new, rates::Nothing)
    if length(J)==1
        QuantumOpticsBase.mul!(psi_new,J[1],psi,true,false)
        psi_new.data ./= norm(psi_new)
        i=1
    else
        probs = zeros(real(eltype(psi)), length(J))
        for i=1:length(J)
            QuantumOpticsBase.mul!(psi_new,J[i],psi,true,false)
            probs[i] = real(dot(psi_new.data, psi_new.data))
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        QuantumOpticsBase.mul!(psi_new,J[i],psi,one(eltype(psi))/sqrt(probs[i]),zero(eltype(psi)))
    end
    return i
end

function jump(rng, t, psi, J, psi_new, rates::AbstractVector)
    if length(J)==1
        QuantumOpticsBase.mul!(psi_new,J[1],psi,eltype(psi)(sqrt(rates[1])),zero(eltype(psi)))
        psi_new.data ./= norm(psi_new)
        i=1
    else
        probs = zeros(real(eltype(psi)), length(J))
        for i=1:length(J)
            QuantumOpticsBase.mul!(psi_new,J[i],psi,eltype(psi)(sqrt(rates[i])),zero(eltype(psi)))
            probs[i] = real(dot(psi_new.data, psi_new.data))
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        QuantumOpticsBase.mul!(psi_new,J[i],psi,eltype(psi)(sqrt(rates[i]/probs[i])),zero(eltype(psi)))
    end
    return i
end

"""
    dmcwf_h!(dpsi, H, J, Jdagger, rates, psi, dpsi_cache)

Update `dpsi` according to a non-hermitian Schrödinger equation. The
non-hermitian Hamiltonian is given in two parts - the hermitian part H and
the jump operators J.

See also: [`mcwf`](@ref)
"""
function dmcwf_h!(dpsi, H, J, Jdagger, rates::Nothing, psi, dpsi_cache)
    QuantumOpticsBase.mul!(dpsi,H,psi,eltype(psi)(-im),zero(eltype(psi)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(dpsi_cache,J[i],psi,true,false)
        QuantumOpticsBase.mul!(dpsi,Jdagger[i],dpsi_cache,eltype(psi)(-0.5),one(eltype(psi)))
    end
    return dpsi
end

function dmcwf_h!(dpsi, H, J, Jdagger, rates::AbstractVector, psi, dpsi_cache)
    QuantumOpticsBase.mul!(dpsi,H,psi,eltype(psi)(-im),zero(eltype(psi)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(dpsi_cache,J[i],psi,eltype(psi)(rates[i]),zero(eltype(psi)))
        QuantumOpticsBase.mul!(dpsi,Jdagger[i],dpsi_cache,eltype(psi)(-0.5),one(eltype(psi)))
    end
    return dpsi
end

"""
    check_mcwf(psi0, H, J, Jdagger, rates)

Check input of mcwf.
"""
function check_mcwf(psi0, H, J, Jdagger, rates)
    # TODO: replace type checks by dispatch; make types of J known
    isreducible = true
    if !(isa(H, DenseOpType) || isa(H, SparseOpType))
        isreducible = false
    end
    for j=J
        @assert isa(j, AbstractOperator)
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator)
        if !(isa(j, DenseOpType) || isa(j, SparseOpType))
            isreducible = false
        end
    end
    @assert length(J) == length(Jdagger)
    if isa(rates, Matrix)
        throw(ArgumentError("Matrix of decay rates not supported for MCWF!
            Use diagonaljumps(rates, J) to calculate new rates and jump operators."))
    elseif isa(rates, Vector)
        @assert length(rates) == length(J)
    end
    isreducible
end

"""
    diagonaljumps(rates, J)

Diagonalize jump operators.

The given matrix `rates` of decay rates is diagonalized and the
corresponding set of jump operators is calculated.

# Arguments
* `rates`: Matrix of decay rates.
* `J`: Vector of jump operators.
"""
function diagonaljumps(rates::AbstractMatrix, J)
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [sum([v[j, i]*J[j] for j=1:length(d)]) for i=1:length(d)]
end

function diagonaljumps(rates::AbstractMatrix, J::Vector{T}) where T<:Union{LazySum,LazyTensor,LazyProduct}
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [LazySum([v[j, i]*J[j] for j=1:length(d)]...) for i=1:length(d)]
end
