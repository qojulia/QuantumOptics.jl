using Random, LinearAlgebra

# TODO: Remove imports
import RecursiveArrayTools.copyat_or_push!

"""
    mcwf_h(tspan, rho0, Hnh, J; <keyword arguments>)

Calculate MCWF trajectory where the Hamiltonian is given in hermitian form.

For more information see: [`mcwf`](@ref)
"""
function mcwf_h(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        tmp::T=copy(psi0),
        display_beforeevent=false, display_afterevent=false,
        kwargs...) where {B<:Basis,T<:Ket{B}}
    check_mcwf(psi0, H, J, Jdagger, rates)
    f(t::Float64, psi::T, dpsi::T) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates)
    j(rng, t::Float64, psi::T, psi_new::T) = jump(rng, t, psi, J, psi_new, rates)
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
function mcwf_nh(tspan, psi0::T, Hnh::AbstractOperator{B,B}, J::Vector;
        seed=rand(UInt), fout=nothing,
        display_beforeevent=false, display_afterevent=false,
        kwargs...) where {B<:Basis,T<:Ket{B}}
    check_mcwf(psi0, Hnh, J, J, nothing)
    f(t::Float64, psi::T, dpsi::T) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t::Float64, psi::T, psi_new::T) = jump(rng, t, psi, J, psi_new, nothing)
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
* `display_jumps=false`: If set to true, an additional list of times and indices
is returned. These correspond to the times at which a jump occured and the index
of the jump operators with which the jump occured, respectively.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf(tspan, psi0::T, H::AbstractOperator{B,B}, J::Vector;
        seed=rand(UInt), rates::DecayRates=nothing,
        fout=nothing, Jdagger::Vector=dagger.(J),
        display_beforeevent=false, display_afterevent=false,
        kwargs...) where {B<:Basis,T<:Ket{B}}
    isreducible = check_mcwf(psi0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(psi0)
        dmcwf_h_(t::Float64, psi::T, dpsi::T) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates)
        j_h(rng, t::Float64, psi::T, psi_new::T) = jump(rng, t, psi, J, psi_new, rates)
        integrate_mcwf(dmcwf_h_, j_h, tspan, psi0, seed,
            fout;
            display_beforeevent=display_beforeevent,
            display_afterevent=display_afterevent,
            kwargs...)
    else
        Hnh = copy(H)
        if typeof(rates) == Nothing
            for i=1:length(J)
                Hnh -= 0.5im*Jdagger[i]*J[i]
            end
        else
            for i=1:length(J)
                Hnh -= 0.5im*rates[i]*Jdagger[i]*J[i]
            end
        end
        dmcwf_nh_(t::Float64, psi::T, dpsi::T) = dmcwf_nh(psi, Hnh, dpsi)
        j_nh(rng, t::Float64, psi::T, psi_new::T) = jump(rng, t, psi, J, psi_new, rates)
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
* `display_jumps=false`: If set to true, an additional list of times and indices
is returned. These correspond to the times at which a jump occured and the index
of the jump operators with which the jump occured, respectively.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf_dynamic(tspan, psi0::T, f::Function;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...) where {T<:Ket}
    tmp = copy(psi0)
    dmcwf_(t::Float64, psi::T, dpsi::T) = dmcwf_h_dynamic(t, psi, f, rates, dpsi, tmp)
    j_(rng, t::Float64, psi::T, psi_new::T) = jump_dynamic(rng, t, psi, f, psi_new, rates)
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
function mcwf_nh_dynamic(tspan, psi0::T, f::Function;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...) where T<:Ket
    dmcwf_(t::Float64, psi::T, dpsi::T) = dmcwf_nh_dynamic(t, psi, f, dpsi)
    j_(rng, t::Float64, psi::T, psi_new::T) = jump_dynamic(rng, t, psi, f, psi_new, rates)
    integrate_mcwf(dmcwf_, j_, tspan, psi0, seed,
        fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

function dmcwf_h_dynamic(t::Float64, psi::T, f::Function, rates::DecayRates,
                    dpsi::T, tmp::T) where T<:Ket
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, rates_)
    dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates_)
end

function dmcwf_nh_dynamic(t::Float64, psi::T, f::Function, dpsi::T) where T<:Ket
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    H, J, Jdagger = result[1:3]
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, nothing)
    dmcwf_nh(psi, H, dpsi)
end

function jump_dynamic(rng, t::Float64, psi::T, f::Function, psi_new::T, rates::DecayRates) where T<:Ket
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
* `kwargs`: Further arguments are passed on to the ode solver.
"""
function integrate_mcwf(dmcwf::Function, jumpfun::Function, tspan,
                        psi0::T, seed, fout::Function;
                        display_beforeevent=false, display_afterevent=false,
                        display_jumps=false,
                        save_everystep=false, callback=nothing,
                        alg=OrdinaryDiffEq.DP5(),
                        kwargs...) where T

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
    jump_t = Float64[]
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

    function fout_(x::Vector{ComplexF64}, t::Float64, integrator)
        recast!(x, state)
        fout(t, state)
    end

    state = copy(psi0)
    dstate = copy(psi0)
    out_type = pure_inference(fout, Tuple{eltype(tspan),typeof(state)})
    out = DiffEqCallbacks.SavedValues(Float64,out_type)
    scb = DiffEqCallbacks.SavingCallback(fout_,out,saveat=tspan,
                                         save_everystep=save_everystep,
                                         save_start = false)

    cb = jump_callback(jumpfun, seed, scb, save_before!, save_after!, save_t_index, psi0)
    full_cb = OrdinaryDiffEq.CallbackSet(callback,cb,scb)

    function df_(dx::D, x::D, p, t) where D<:Vector{ComplexF64}
        recast!(x, state)
        recast!(dx, dstate)
        dmcwf(t, state, dstate)
        recast!(dstate, dx)
    end

    prob = OrdinaryDiffEq.ODEProblem{true}(df_, as_vector(psi0),(tspan[1],tspan[end]))

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

function integrate_mcwf(dmcwf::Function, jumpfun::Function, tspan,
                        psi0::T, seed, fout::Nothing;
                        kwargs...) where T
    function fout_(t::Float64, x::T)
        return normalize(x)
    end
    integrate_mcwf(dmcwf, jumpfun, tspan, psi0, seed, fout_; kwargs...)
end

function jump_callback(jumpfun::Function, seed, scb, save_before!::Function,
                        save_after!::Function, save_t_index::Function, psi0::Ket)

    tmp = copy(psi0)
    psi_tmp = copy(psi0)

    rng = MersenneTwister(convert(UInt, seed))
    jumpnorm = Ref(rand(rng))
    djumpnorm(x::Vector{ComplexF64}, t::Float64, integrator) = norm(x)^2 - (1-jumpnorm[])

    function dojump(integrator)
        x = integrator.u
        t = integrator.t

        affect! = scb.affect!
        save_before!(affect!,integrator)
        recast!(x, psi_tmp)
        i = jumpfun(rng, t, psi_tmp, tmp)
        x .= tmp.data
        save_after!(affect!,integrator)
        save_t_index(t,i)

        jumpnorm[] = rand(rng)
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
function jump(rng, t::Float64, psi::T, J::Vector, psi_new::T, rates::Nothing) where T<:Ket
    if length(J)==1
        QuantumOpticsBase.gemv!(complex(1.), J[1], psi, complex(0.), psi_new)
        psi_new.data ./= norm(psi_new)
        i=1
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            QuantumOpticsBase.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
            probs[i] = dot(psi_new.data, psi_new.data)
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        QuantumOpticsBase.gemv!(complex(1.)/sqrt(probs[i]), J[i], psi, complex(0.), psi_new)
    end
    return i
end

function jump(rng, t::Float64, psi::T, J::Vector, psi_new::T, rates::Vector{Float64}) where T<:Ket
    if length(J)==1
        QuantumOpticsBase.gemv!(complex(sqrt(rates[1])), J[1], psi, complex(0.), psi_new)
        psi_new.data ./= norm(psi_new)
        i=1
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            QuantumOpticsBase.gemv!(complex(sqrt(rates[i])), J[i], psi, complex(0.), psi_new)
            probs[i] = dot(psi_new.data, psi_new.data)
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        QuantumOpticsBase.gemv!(complex(sqrt(rates[i]/probs[i])), J[i], psi, complex(0.), psi_new)
    end
    return i
end

"""
Evaluate non-hermitian Schroedinger equation.

The non-hermitian Hamiltonian is given in two parts - the hermitian part H and
the jump operators J.
"""
function dmcwf_h(psi::T, H::AbstractOperator{B,B},
                 J::Vector, Jdagger::Vector, dpsi::T, tmp::T, rates::Nothing) where {B<:Basis,T<:Ket{B}}
    QuantumOpticsBase.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    for i=1:length(J)
        QuantumOpticsBase.gemv!(complex(1.), J[i], psi, complex(0.), tmp)
        QuantumOpticsBase.gemv!(-complex(0.5,0.), Jdagger[i], tmp, complex(1.), dpsi)
    end
    return dpsi
end

function dmcwf_h(psi::T, H::AbstractOperator{B,B},
                 J::Vector, Jdagger::Vector, dpsi::T, tmp::T, rates::Vector{Float64}) where {B<:Basis,T<:Ket{B}}
    QuantumOpticsBase.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    for i=1:length(J)
        QuantumOpticsBase.gemv!(complex(rates[i]), J[i], psi, complex(0.), tmp)
        QuantumOpticsBase.gemv!(-complex(0.5,0.), Jdagger[i], tmp, complex(1.), dpsi)
    end
    return dpsi
end


"""
Evaluate non-hermitian Schroedinger equation.

The given Hamiltonian is already the non-hermitian version.
"""
function dmcwf_nh(psi::T, Hnh::AbstractOperator{B,B}, dpsi::T) where {B<:Basis,T<:Ket{B}}
    QuantumOpticsBase.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return dpsi
end

"""
    check_mcwf(psi0, H, J, Jdagger, rates)

Check input of mcwf.
"""
function check_mcwf(psi0::Ket{B}, H::AbstractOperator{B,B}, J::Vector, Jdagger::Vector, rates::DecayRates) where B<:Basis
    # TODO: replace type checks by dispatch; make types of J known
    isreducible = true
    if !(isa(H, DenseOperator) || isa(H, SparseOperator))
        isreducible = false
    end
    for j=J
        @assert isa(j, AbstractOperator{B,B})
        if !(isa(j, DenseOperator) || isa(j, SparseOperator))
            isreducible = false
        end
    end
    for j=Jdagger
        @assert isa(j, AbstractOperator{B,B})
        if !(isa(j, DenseOperator) || isa(j, SparseOperator))
            isreducible = false
        end
    end
    @assert length(J) == length(Jdagger)
    if typeof(rates) == Matrix{Float64}
        throw(ArgumentError("Matrix of decay rates not supported for MCWF!
            Use diagonaljumps(rates, J) to calculate new rates and jump operators."))
    elseif typeof(rates) == Vector{Float64}
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
function diagonaljumps(rates::Matrix{Float64}, J::Vector{T}) where {B<:Basis,T<:AbstractOperator{B,B}}
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [sum([v[j, i]*J[j] for j=1:length(d)]) for i=1:length(d)]
end

function diagonaljumps(rates::Matrix{Float64}, J::Vector{T}) where {B<:Basis,T<:Union{LazySum{B,B},LazyTensor{B,B},LazyProduct{B,B}}}
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [LazySum([v[j, i]*J[j] for j=1:length(d)]...) for i=1:length(d)]
end
