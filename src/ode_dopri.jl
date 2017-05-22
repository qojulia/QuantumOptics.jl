module ode_dopri

export ode, ode_event

using Roots

const order = 5

# Runge-Kutta table
const a2 = Float64[1/5]
const a3 = Float64[3/40, 9/40]
const a4 = Float64[44/45, -56/15, 32/9]
const a5 = Float64[19372/6561, -25360/2187, 64448/6561, -212/729]
const a6 = Float64[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
const a7 = Float64[35/384, 0., 500/1113, 125/192, -2187/6784, 11/84]
const a = Vector{Float64}[Float64[], a2, a3, a4, a5, a6, a7]

# Weights
const bs = Float64[5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

# Nodes
const c = Float64[0., 1/5, 3/10, 4/5, 8/9, 1., 1.]

b7(θ) = (θ^2*(θ-1) + θ^2*(θ-1)^2*10*(7414447 - 829305*θ)/29380423)
b1(θ) = θ^2*(3-2*θ)*a7[1] + θ*(θ-1)^2 - θ^2*(θ-1)^2*5*(2558722523-31403016*θ)/11282082432
b3(θ) = θ^2*(3-2*θ)*a7[3] + θ^2*(θ-1)^2 * 100*(882725551 - 15701508*θ)/32700410799
b4(θ) = θ^2*(3-2*θ)*a7[4] - θ^2*(θ-1)^2 * 25*(443332067 - 31403016*θ)/1880347072
b5(θ) = θ^2*(3-2*θ)*a7[5] + θ^2*(θ-1)^2 * 32805*(23143187 - 3489224*θ)/199316789632
b6(θ) = θ^2*(3-2*θ)*a7[6] - θ^2*(θ-1)^2 * 55*(29972135 - 7076736*θ)/822651844


"""
Intermediate function value for the nth step.

Arguments:
----------
    * x: End value.
    * x0: Starting value.
    * h: Step size.
    * coeffs: Runge-kutta coefficents for the nth substep.
    * k: Function derivatives at the previous substep points.
"""
function substep{T}(x::Vector{T}, x0::Vector{T}, h::Float64, coeffs::Vector{Float64}, k::Vector{Vector{T}})
    @inbounds for m=1:length(x0)
        dx::T = 0.
        @inbounds for i=1:length(coeffs)
            dx += coeffs[i] * k[i][m]
        end
        x[m] = x0[m] + h*dx
    end
    return nothing
end


"""
Perform one single Runge-Kutta step.

Arguments:
----------
    * F: Derivative function with signature F(t, y, dy) which writes the derivative into dy.
    * t0: Start time of the step.
    * h: Length of the step.
    * x0: Initial state.
    * xp: Final state after the Runge-Kutta step. (5th order)
    * xs: Final state after the Runge-Kutta step. (4th order)
    * k: States at the substeps.
"""
function step{T}(F::Function, t0::Float64, h::Float64,
                x0::Vector{T}, xp::Vector{T}, xs::Vector{T}, k::Vector{Vector{T}})
    for i=2:length(c)
        substep(xp, x0, h, a[i], k)
        F(t0 + h*c[i], xp, k[i])
    end
    substep(xs, x0, h, bs, k)
    return nothing
end


"""
Allocate necessary memory.
"""
function allocate_memory{T}(x::Vector{T})
    xp = zeros(T, length(x))
    xs = zeros(T, length(x))
    k = Vector{T}[]
    for i=1:7
        push!(k, zeros(T, length(x)))
    end
    return (xp, xs, k)
end


"""
Calculate state for intermediate steps (Dense output).

Arguments:
----------
    * t0: Start time of the step.
    * x0: Initial state.
    * h: Length of the step.
    * k: States at the substeps.
    * t: Time at which the state should be interpolated.
    * x: Resulting state at the given point of time.
"""
function interpolate{T}(t0::Float64, x0::Vector{T}, h::Float64, k::Vector{Vector{T}}, t::Float64, x::Vector{T})
    θ = (t-t0)/h
    b1_ = b1(θ); b3_ = b3(θ); b4_ = b4(θ); b5_ = b5(θ); b6_ = b6(θ); b7_ = b7(θ)
    for i=1:length(x0)
        dx = b1_*k[1][i] + b3_*k[3][i] + b4_*k[4][i]
        dx += b5_*k[5][i] + b6_*k[6][i] + b7_*k[7][i]
        x[i] = x0[i] + h*dx
    end
    nothing
end


"""
Estimate the error by comparing results of different order Runge-Kutta methods.
"""
function error_estimate(xp, xs, abstol, reltol)
    err::Float64 = 0.
    for i=1:length(xp)
        sc_i = abstol + reltol*max(abs(xp[i]), abs(xs[i]))
        err += abs2(xp[i]-xs[i])/sc_i^2
    end
    return sqrt(err/length(xp))
end


"""
Find a good guess for an initial time step.
"""
function initial_stepsize(F, t, x, k, abstol, reltol, tmp1, tmp2)
    d0 = 0.
    d1 = 0.
    for i=1:length(x)
        sc_i2 = (abstol + abs(x[i])*reltol)^2
        d0 += abs2(x[i])/sc_i2
        d1 += abs2(k[1][i])/sc_i2
    end
    d0 = sqrt(d0/length(x))
    d1 = sqrt(d1/length(x))
    h0 = ((d0<1e-5 || d1<1e-5) ? 1e-6 : 0.01*d0/d1)
    substep(tmp1, x, h0, [1.], k)
    F(t+h0, tmp1, tmp2)
    for i=1:length(tmp2)
        tmp2[i] -= k[1][i]
    end
    d2 = norm(tmp2, 2)
    if max(d1,d2)<1e-15
        h1 = max(1e-6, h0*1e-3)
    else
        h1 = (0.01/max(d1,d2))^(1./order)
    end
    return min(100*h0, h1)
end


"""
Determine the size of the next step.
"""
function stepsize_strategy(err, laststepaccepted, h, hmin, hmax)
    accept_step = err<1
    facmin = (laststepaccepted ? 5. : 1.)
    hnew = h*min(facmin, max(0.2, 0.9*(1./err)^(1./order)))
    hnew = min(hmax, hnew)
    if hnew<hmin
        error("Stepsize below hmin.")
    end
    return hnew, accept_step
end


"""
Display states at the given times which are in the current Runge-Kutta step.
"""
function display_steps{T}(fout::Function, tspan::Vector{Float64}, t::Float64, x::Vector{T}, h::Float64, k::Vector{Vector{T}}, xs::Vector{T})
    for tout=tspan
        if t<tout<=t+h
            interpolate(t, x, h, k, tout, xs)
            fout(tout, xs)
        end
    end
end


@enum CallbackCommand nojump jump stop


"""
Adaptive Runge-Kutta Dormand-Prince 4(5) solver with event handling and dense output.

Arguments:
----------
    * F: Derivative function with signature F(t, y, dy) which writes the derivative into dy.
    * tspan: Vector of times at which output should be displayd.
    * x0: Initial state.
    * event_locator: Function used to find events with signature
            event_locator(t, x) returning a real value. If the sign of the returned value
            changes the event_callback function is called.
    * event_callback: Function that is called when an event happens. Its
            signature is event_callback(t, x) and it should return a CallbackCommand.
            The possible CallBack commands are:
                - nojump: No changes in the dynamics. In this case x should not be changed
                          inside the callback function.
                - jump: the x vector has changed and time evolution continues from t_event.
                - stop: The ode solver stops ia the event time.

Optional Arguments:
-------------------
    * fout: Function called to display the state at the given points of time
            with signature fout(t, x). If no function is given, ode_event returns
            a vector with the states at all points in time given in tspan.
    * reltol: Relative error tolerance.
    * abstol: Absolute error tolerance.
    * h0: Initial guess for the size of the time step. If no number is given an
            initial timestep is chosen automatically.
    * hmin: If the automatic stepsize goes below this limit the ode solver stops
            with an error.
    * hmax: Stepsize is never increased above this limit.
    * display_initialvalue: Call fout function at tspan[1].
    * display_finalvalue: Call fout function at tspan[end].
    * display_intermediatesteps: Call fout function after every Runge-Kutta step.
    * display_beforeevent: Call fout function immediately before an event.
    * display_afterevent: Call fout function immediately after an event.
"""
function ode_event{T}(F, tspan::Vector{Float64}, x0::Vector{T}, fout::Function,
                    event_locator::Function, event_callback::Function;
                    reltol::Float64 = 1.0e-6,
                    abstol::Float64 = 1.0e-8,
                    h0::Float64 = NaN,
                    hmin::Float64 = (tspan[end]-tspan[1])/1e9,
                    hmax::Float64 = (tspan[end]-tspan[1]),
                    display_initialvalue::Bool = true,
                    display_finalvalue::Bool = true,
                    display_intermediatesteps::Bool = false,
                    display_beforeevent::Bool = false,
                    display_afterevent::Bool = false
                    )
    t, tfinal = tspan[1], tspan[end]
    display_initialvalue && fout(t, x0)

    # Allocate memory
    x = copy(x0) # Dont change initial state.
    xp, xs, k = allocate_memory(x0)

    # Initial derivative.
    F(t,x,k[1])

    # Find initial step size.
    h = (h0===NaN ? initial_stepsize(F, t, x, k, abstol, reltol, k[2], k[3]) : h0)
    h = max(hmin, h)
    h = min(hmax, h)

    accept_step = true
    while t < tfinal
        # Try Runge-Kutta steps until the error is below the tolerances.
        step(F, t, h, x, xp, xs, k)
        err = error_estimate(xp, xs, abstol, reltol)
        hnew, accept_step = stepsize_strategy(err, accept_step, h, hmin, hmax)
        if accept_step
            # Check if there is an event in the last time step.
            e1 = event_locator(t+hmin,x)
            e2 = event_locator(t+h,xp)
            if e2==0. || e1*e2 < 0. # Handle event case.
                t_event = fzero(t_->(interpolate(t, x, h, k, t_, xs); event_locator(t_, xs)), t, t+h)
                display_steps(fout, tspan[tspan.<t_event], t, x, h, k, xs)
                interpolate(t, x, h, k, t_event, xs)
                display_beforeevent && fout(t_event, xs)
                cmd = event_callback(t_event, xs)
                if typeof(cmd)!=CallbackCommand
                    error("Event callback function has to return a CallbackCommand.")
                end
                display_afterevent && fout(t_event, xs)
                if cmd == stop
                    return nothing
                elseif cmd == jump
                    F(t,x,k[1])
                    h = 0.9*hnew
                    xs, x = x, xs
                    t = t_event
                    continue
                elseif cmd == nojump
                    display_steps(fout, tspan[tspan.>t_event], t, x, h, k, xs)
                    display_intermediatesteps && fout(t, xp)
                    xp, x = x, xp
                    k[1], k[end] = k[end], k[1]
                    t = t + h
                else
                    error("Unrecognized event command.")
                end
            else # Handle no-event case.
                display_steps(fout, tspan, t, x, h, k, xs)
                if display_intermediatesteps && t+h<tfinal
                    fout(t+h, xp)
                end
                xp, x = x, xp
                k[1], k[end] = k[end], k[1]
                t = t + h
            end
        end
        h = hnew
    end
end


function ode_event{T}(F, tspan::Vector{Float64}, x0::Vector{T},
                    event_locator::Function, event_callback::Function;
                    args...)
    tout = Float64[]
    xout = Vector{T}[]
    fout = (t, x) -> (push!(tout, t); push!(xout, copy(x)); nothing)
    ode_event(F, tspan, x0, fout, event_locator, event_callback; args...)
    return tout, xout
end


"""
Adaptive Runge-Kutta Dormand-Prince 4(5) solver with dense output.

Arguments:
----------
    * F: Derivative function with signature F(t, y, dy) which writes the derivative into dy.
    * tspan: Vector of times at which output should be displayd.
    * x0: Initial state.

Optional Arguments:
-------------------
    * fout: Function called to display the state at the given points of time
            with signature fout(t, x). If no function is given, ode_event returns
            a vector with the states at all points in time given in tspan.
    * reltol: Relative error tolerance.
    * abstol: Absolute error tolerance.
    * h0: Initial guess for the size of the time step. If no number is given an
            initial timestep is chosen automatically.
    * hmin: If the automatic stepsize goes below this limit the ode solver stops
            with an error.
    * hmax: Stepsize is never increased above this limit.
    * display_initialvalue: Call fout function at tspan[1].
    * display_finalvalue: Call fout function at tspan[end].
    * display_intermediatesteps: Call fout function after every Runge-Kutta step.
"""
function ode{T}(F, tspan::Vector{Float64}, x0::Vector{T}, fout::Function;
                    reltol::Float64 = 1.0e-6,
                    abstol::Float64 = 1.0e-8,
                    h0::Float64 = NaN,
                    hmin::Float64 = (tspan[end]-tspan[1])/1e9,
                    hmax::Float64 = (tspan[end]-tspan[1]),
                    display_initialvalue::Bool = true,
                    display_finalvalue::Bool = true,
                    display_intermediatesteps::Bool = false,
                    )
    t, tfinal = tspan[1], tspan[end]
    display_initialvalue && fout(t, x0)

    # Allocate memory
    x = copy(x0) # Dont change initial state.
    xp, xs, k = allocate_memory(x0)

    # Initial derivative.
    F(t,x,k[1])

    # Find initial step size.
    h = (h0===NaN ? initial_stepsize(F, t, x, k, abstol, reltol, k[2], k[3]) : h0)
    h = max(hmin, h)
    h = min(hmax, h)

    accept_step = true
    while t < tfinal
        # Try Runge-Kutta steps until the error is below the tolerances.
        step(F, t, h, x, xp, xs, k)
        err = error_estimate(xp, xs, abstol, reltol)
        hnew, accept_step = stepsize_strategy(err, accept_step, h, hmin, hmax)
        if accept_step
            display_steps(fout, tspan, t, x, h, k, xs)
            if display_intermediatesteps && t+h<tfinal
                fout(t+h, xp)
            end
            xp, x = x, xp
            k[1], k[end] = k[end], k[1]
            t = t + h
        end
        h = hnew
    end
end


function ode{T}(F, tspan::Vector{Float64}, x0::Vector{T}; args...)
    tout = Float64[]
    xout = Vector{T}[]
    fout = (t, x) -> (push!(tout, t); push!(xout, copy(x)); nothing)
    ode(F, tspan, x0, fout; args...)
    return tout, xout
end


end # module
