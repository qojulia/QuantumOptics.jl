module timeevolution_mcwf

export mcwf, mcwf_h, mcwf_nh, mcwf_dynamic, mcwf_nh_dynamic, diagonaljumps

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse
using ..timeevolution
using ...operators_lazysum, ...operators_lazytensor, ...operators_lazyproduct
using Random, LinearAlgebra
import OrdinaryDiffEq

# TODO: Remove imports
import DiffEqCallbacks, RecursiveArrayTools.copyat_or_push!
import ..recast!, ..QO_CHECKS
Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

const DecayRates = Union{Vector{Float64}, Matrix{Float64}, Nothing}

"""
    mcwf_h(tspan, rho0, Hnh, J; <keyword arguments>)

Calculate MCWF trajectory where the Hamiltonian is given in hermitian form.

For more information see: [`mcwf`](@ref)
"""
function mcwf_h(tspan, psi0::Ket, H::AbstractOperator, J::Vector;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, Jdagger::Vector=dagger.(J),
    tmp::Ket=copy(psi0),
    display_beforeevent=false, display_afterevent=false,
    kwargs...)
    check_mcwf(psi0, H, J, Jdagger, rates)
    f(t, psi, dpsi) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
    return integrate_mcwf(f, j, tspan, psi0, seed, fout;
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
function mcwf_nh(tspan, psi0::Ket{B}, Hnh::AbstractOperator{B,B}, J::Vector{T};
    seed=rand(UInt), fout=nothing,
    display_beforeevent=false, display_afterevent=false,
    kwargs...) where {B<:Basis,T<:AbstractOperator{B,B}}
    check_mcwf(psi0, Hnh, J, J, nothing)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, nothing)
    return integrate_mcwf(f, j, tspan, psi0, seed, fout;
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
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf(tspan, psi0::Ket, H::AbstractOperator, J::Vector;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, Jdagger::Vector=dagger.(J),
    display_beforeevent=false, display_afterevent=false,
    kwargs...)
    isreducible = check_mcwf(psi0, H, J, Jdagger, rates)
    if !isreducible
        tmp = copy(psi0)
        dmcwf_h_(t, psi, dpsi) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates)
        j_h(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
        return integrate_mcwf(dmcwf_h_, j_h, tspan, psi0, seed,
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
        dmcwf_nh_(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
        j_nh(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new, rates)
        return integrate_mcwf(dmcwf_nh_, j_nh, tspan, psi0, seed,
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
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function mcwf_dynamic(tspan, psi0::Ket, f::Function;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...)
    tmp = copy(psi0)
    dmcwf_(t, psi, dpsi) = dmcwf_h_dynamic(t, psi, f, rates, dpsi, tmp)
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
function mcwf_nh_dynamic(tspan, psi0::Ket, f::Function;
    seed=rand(UInt), rates::DecayRates=nothing,
    fout=nothing, display_beforeevent=false, display_afterevent=false,
    kwargs...)
    dmcwf_(t, psi, dpsi) = dmcwf_nh_dynamic(t, psi, f, dpsi)
    j_(rng, t, psi, psi_new) = jump_dynamic(rng, t, psi, f, psi_new, rates)
    integrate_mcwf(dmcwf_, j_, tspan, psi0, seed,
        fout;
        display_beforeevent=display_beforeevent,
        display_afterevent=display_afterevent,
        kwargs...)
end

function dmcwf_h_dynamic(t::Float64, psi::Ket, f::Function, rates::DecayRates,
                    dpsi::Ket, tmp::Ket)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    if length(result) == 3
        H, J, Jdagger = result
        rates_ = rates
    else
        H, J, Jdagger, rates_ = result
    end
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, rates_)
    dmcwf_h(psi, H, J, Jdagger, dpsi, tmp, rates)
end

function dmcwf_nh_dynamic(t::Float64, psi::Ket, f::Function, dpsi::Ket)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    H, J, Jdagger = result[1:3]
    QO_CHECKS[] && check_mcwf(psi, H, J, Jdagger, nothing)
    dmcwf_nh(psi, H, dpsi)
end

function jump_dynamic(rng, t::Float64, psi::Ket, f::Function, psi_new::Ket, rates::DecayRates)
    result = f(t, psi)
    QO_CHECKS[] && @assert 3 <= length(result) <= 4
    J = result[2]
    if length(result) == 3
        rates_ = rates
    else
        rates_ = result[4]
    end
    jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket, rates::DecayRates)
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
                        psi0::Ket, seed, fout::Function;
                        display_beforeevent=false, display_afterevent=false,
                        #TODO: Remove kwargs
                        save_everystep=false, callback=nothing,
                        alg=OrdinaryDiffEq.DP5(),
                        kwargs...)

    tmp = copy(psi0)
    as_ket(x::Vector{ComplexF64}) = Ket(psi0.basis, x)
    as_vector(psi::Ket) = psi.data
    rng = MersenneTwister(convert(UInt, seed))
    jumpnorm = Ref(rand(rng))
    djumpnorm(x::Vector{ComplexF64}, t, integrator) = norm(as_ket(x))^2 - (1-jumpnorm[])

    if !display_beforeevent && !display_afterevent
        function dojump(integrator)
            x = integrator.u
            t = integrator.t
            jumpfun(rng, t, as_ket(x), tmp)
            x .= tmp.data
            jumpnorm[] = rand(rng)
        end
        cb = OrdinaryDiffEq.ContinuousCallback(djumpnorm,dojump,
                         save_positions = (display_beforeevent,display_afterevent))


        return timeevolution.integrate(float(tspan), dmcwf, as_vector(psi0),
                    copy(psi0), copy(psi0), fout;
                    callback = cb,
                    kwargs...)
    else
        # Temporary workaround until proper tooling for saving
        # TODO: Replace by proper call to timeevolution.integrate
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

        function dojump_display(integrator)
            x = integrator.u
            t = integrator.t

            affect! = scb.affect!
            if display_beforeevent
                affect!.saveiter += 1
                copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
                copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                    affect!.save_func(integrator.u, integrator.t, integrator),Val{false})
            end

            jumpfun(rng, t, as_ket(x), tmp)

            if display_afterevent
                affect!.saveiter += 1
                copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
                copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                    affect!.save_func(integrator.u, integrator.t, integrator),Val{false})
            end

            x .= tmp.data
            jumpnorm[] = rand(rng)
        end

        cb = OrdinaryDiffEq.ContinuousCallback(djumpnorm,dojump_display,
                         save_positions = (false,false))
        full_cb = OrdinaryDiffEq.CallbackSet(callback,cb,scb)

        function df_(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, p, t)
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
        return out.t, out.saveval
    end
end

function integrate_mcwf(dmcwf::Function, jumpfun::Function, tspan,
                        psi0::Ket, seed, fout::Nothing;
                        kwargs...)
    function fout_(t, x)
        psi = copy(x)
        psi /= norm(psi)
        return psi
    end
    integrate_mcwf(dmcwf, jumpfun, tspan, psi0, seed, fout_; kwargs...)
end

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
function jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket, rates::Nothing)
    if length(J)==1
        mul!(psi_new, J[1], psi)
        psi_new.data ./= norm(psi_new)
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            mul!(psi_new, J[i], psi)
            probs[i] = dot(psi_new.data, psi_new.data)
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        mul!(psi_new, J[i], psi, 1.0/sqrt(probs[i]), 0.0)
    end
    return nothing
end

function jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket, rates::Vector{Float64})
    if length(J)==1
        mul!(psi_new, J[1], psi, sqrt(rates[1]), 0.0)
        psi_new.data ./= norm(psi_new)
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            mul!(psi_new, J[i], psi, sqrt(rates[i]), 0.0)
            probs[i] = dot(psi_new.data, psi_new.data)
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        mul!(psi_new, J[i], psi, sqrt(rates[i]/probs[i]), 0.0)
    end
    return nothing
end

"""
Evaluate non-hermitian Schroedinger equation.

The non-hermitian Hamiltonian is given in two parts - the hermitian part H and
the jump operators J.
"""
function dmcwf_h(psi::Ket, H::AbstractOperator,
                 J::Vector, Jdagger::Vector, dpsi::Ket, tmp::Ket, rates::Nothing)
    mul!(dpsi, H, psi, -1.0im, 0.0)
    for i=1:length(J)
        mul!(tmp, J[i], psi)
        mul!(dpsi, Jdagger[i], tmp, -0.5, 1.0)
    end
    return dpsi
end

function dmcwf_h(psi::Ket, H::AbstractOperator,
                 J::Vector, Jdagger::Vector, dpsi::Ket, tmp::Ket, rates::Vector{Float64})
    mul!(dpsi, H, psi, -1.0im, 0.0)
    for i=1:length(J)
        mul!(tmp, J[i], psi, rates[i], 0.0)
        mul!(dpsi, Jdagger[i], tmp, -0.5, 1.0)
    end
    return dpsi
end


"""
Evaluate non-hermitian Schroedinger equation.

The given Hamiltonian is already the non-hermitian version.
"""
function dmcwf_nh(psi::Ket, Hnh::AbstractOperator, dpsi::Ket)
    mul!(dpsi, Hnh, psi, -1.0im, 0.0)
    return dpsi
end

"""
    check_mcwf(psi0, H, J, Jdagger, rates)

Check input of mcwf.
"""
function check_mcwf(psi0::Ket{B}, H::AbstractOperator{B,B}, J::Vector{T},
            Jdagger::Vector{T}, rates::DecayRates) where {B<:Basis,T<:AbstractOperator{B,B}}
    isreducible = true
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
function diagonaljumps(rates::Matrix{Float64}, J::Vector{T}) where T <: AbstractOperator
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [sum([v[j, i]*J[j] for j=1:length(d)]) for i=1:length(d)]
end

function diagonaljumps(rates::Matrix{Float64}, J::Vector{T}) where T <: Union{LazySum, LazyTensor, LazyProduct}
    @assert length(J) == size(rates)[1] == size(rates)[2]
    d, v = eigen(rates)
    d, [LazySum([v[j, i]*J[j] for j=1:length(d)]...) for i=1:length(d)]
end

end #module
