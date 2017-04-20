module timeevolution_mcwf

export mcwf, mcwf_h, mcwf_nh, diagonaljumps

using ...bases
using ...states
using ...operators
using ...ode_dopri

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
function integrate_mcwf(dmcwf::Function, jumpfun::Function, tspan, psi0::Ket, seed;
                fout=nothing,
                kwargs...)
    tmp = deepcopy(psi0)
    as_ket(x::Vector{Complex128}) = Ket(psi0.basis, x)
    as_vector(psi::Ket) = psi.data
    rng = MersenneTwister(convert(UInt, seed))
    jumpnorm = Float64[rand(rng)]
    djumpnorm(t, x::Vector{Complex128}) = norm(as_ket(x))^2 - (1-jumpnorm[1])
    function dojump(t, x::Vector{Complex128})
        jumpfun(rng, t, as_ket(x), tmp)
        for i=1:length(x)
            x[i] = tmp.data[i]
        end
        jumpnorm[1] = rand(rng)
        return ode_dopri.jump
    end

    tout = Float64[]
    xout = Ket[]
    function fout_(t, x::Vector{Complex128})
        if fout==nothing
            psi = deepcopy(as_ket(x))
            psi /= norm(psi)
            push!(tout, t)
            push!(xout, psi)
            return nothing
        else
            return fout(t, as_ket(x))
        end
    end
    dmcwf_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmcwf(t, as_ket(x), as_ket(dx))
    ode_event(dmcwf_, float(tspan), as_vector(psi0), fout_,
        djumpnorm, dojump;
        kwargs...)
    return fout==nothing ? (tout, xout) : nothing
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
function jump(rng, t::Float64, psi::Ket, J::Vector, psi_new::Ket)
    if length(J)==1
        operators.gemv!(complex(1.), J[1], psi, complex(0.), psi_new)
        N = norm(psi_new)
        for i=1:length(psi_new.data)
            psi_new.data[i] /= N
        end
    else
        probs = zeros(Float64, length(J))
        for i=1:length(J)
            operators.gemv!(complex(1.), J[i], psi, complex(0.), psi_new)
            #probs[i] = norm(psi_new)^2
            probs[i] = dagger(psi_new)*psi_new
        end
        cumprobs = cumsum(probs./sum(probs))
        r = rand(rng)
        i = findfirst(cumprobs.>r)
        operators.gemv!(complex(1.)/sqrt(probs[i]), J[i], psi, complex(0.), psi_new)
    end
    return nothing
end

"""
Evaluate non-hermitian Schroedinger equation.

The non-hermitian Hamiltonian is given in two parts - the hermitian part H and
the jump operators J.
"""
function dmcwf_h(psi::Ket, H::Operator,
                 J::Vector, Jdagger::Vector, dpsi::Ket, tmp::Ket)
    operators.gemv!(complex(0,-1.), H, psi, complex(0.), dpsi)
    for i=1:length(J)
        operators.gemv!(complex(1.), J[i], psi, complex(0.), tmp)
        operators.gemv!(-complex(0.5,0.), Jdagger[i], tmp, complex(1.), dpsi)
    end
    return dpsi
end


"""
Evaluate non-hermitian Schroedinger equation.

The given Hamiltonian is already the non-hermitian version.
"""
function dmcwf_nh(psi::Ket, Hnh::Operator, dpsi::Ket)
    operators.gemv!(complex(0,-1.), Hnh, psi, complex(0.), dpsi)
    return dpsi
end

"""
    mcwf_h(tspan, rho0, Hnh, J; <keyword arguments>)

Calculate MCWF trajectory where the Hamiltonian is given in hermitian form.

For more information see: [`mcwf`](@ref)
"""
function mcwf_h(tspan, psi0::Ket, H::Operator, J::Vector;
                seed=rand(UInt), fout=nothing, Jdagger::Vector=map(dagger, J),
                tmp::Ket=deepcopy(psi0),
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    f(t, psi, dpsi) = dmcwf_h(psi, H, J, Jdagger, dpsi, tmp)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout,
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
function mcwf_nh(tspan, psi0::Ket, Hnh::Operator, J::Vector;
                seed=rand(UInt), fout=nothing,
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed; fout=fout,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...)
end

"""
    mcwf(tspan, rho0, H, J; <keyword arguments>)

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
function mcwf(tspan, psi0::Ket, H::Operator, J::Vector;
                seed=rand(UInt), fout=nothing, Jdagger::Vector=map(dagger, J),
                display_beforeevent=false, display_afterevent=false,
                kwargs...)
    Hnh = deepcopy(H)
    for i=1:length(J)
        Hnh -= 0.5im*Jdagger[i]*J[i]
    end
    f(t, psi, dpsi) = dmcwf_nh(psi, Hnh, dpsi)
    j(rng, t, psi, psi_new) = jump(rng, t, psi, J, psi_new)
    return integrate_mcwf(f, j, tspan, psi0, seed;
                fout=fout,
                display_beforeevent=display_beforeevent,
                display_afterevent=display_afterevent,
                kwargs...)
end

"""
    diagonaljumps(Gamma, J)

Diagonalize jump operators.

The given matrix `Gamma` of decay rates is diagonalized and the
corresponding set of jump operators is calculated.
"""
function diagonaljumps(Gamma::Array{Float64}, J::Vector)
  d, v = eig(Gamma)
  d, [sum([v[j, i]*J[j] for j=1:length(d)]) for i=1:length(d)]
end


end #module
