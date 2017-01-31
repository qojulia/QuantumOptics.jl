module correlations

using ..operators
using ..operators_dense
using ..timeevolution
using ..metrics
using ..steadystate


"""
Calculate two time correlation values :math:`\\langle A(t) B(0) \\rangle`

The calculation is done by multiplying the initial density operator
with :math:`B` performing a time evolution according to a master equation
and then calculating the expectation value :math:`\\mathrm{Tr} \\{ A \\rho \\}`

Arguments
---------

tspan
    Points of time at which the correlation should be calculated.
rho0
    Initial density operator.
H
    Operator specifying the Hamiltonian.
J
    Vector of jump operators.
op1
    Operator at time t.
op2
    Operator at time t=0.


Keyword Arguments
-----------------

Gamma
    Vector or matrix specifying the coefficients for the jump operators.
Jdagger (optional)
    Vector containing the hermitian conjugates of the jump operators. If they
    are not given they are calculated automatically.
kwargs
    Further arguments are passed on to the ode solver.
"""
function correlation(tspan::Vector{Float64}, rho0::DenseOperator, H::Operator, J::Vector,
                     op1::Operator, op2::Operator;
                     Gamma::Union{Real, Vector, Matrix}=ones(Float64, length(J)),
                     Jdagger::Vector=map(dagger, J),
                     tmp::DenseOperator=deepcopy(rho0),
                     kwargs...)
    exp_values = Complex128[]
    function fout(t, rho)
        push!(exp_values, expect(op1, rho))
    end
    timeevolution.master(tspan, op2*rho0, H, J; Gamma=Gamma, Jdagger=Jdagger,
                        tmp=tmp, fout=fout, kwargs...)
    return exp_values
end


"""
Calculate two time correlation values :math:`\\langle A(t) B(0) \\rangle`

The calculation is done by multiplying the initial density operator
with :math:`B` performing a time evolution according to a master equation
and then calculating the expectation value :math:`\\mathrm{Tr} \\{ A \\rho \\}`.
The points of time are chosen automatically from the ode solver and the final
time is determined by the steady state termination criterion specified in
:func:`steadystate.master`.

Arguments
---------

rho0
    Initial density operator.
H
    Operator specifying the Hamiltonian.
J
    Vector of jump operators.
op1
    Operator at time t.
op2
    Operator at time t=0.


Keyword Arguments
-----------------

eps
    Tracedistance used as termination criterion.
h0
    Initial time step used in the time evolution.
Gamma
    Vector or matrix specifying the coefficients for the jump operators.
Jdagger (optional)
    Vector containing the hermitian conjugates of the jump operators. If they
    are not given they are calculated automatically.
kwargs
    Further arguments are passed on to the ode solver.
"""
function correlation(rho0::DenseOperator, H::Operator, J::Vector,
                     op1::Operator, op2::Operator;
                     eps::Float64=1e-4, h0=10.,
                     Gamma::Union{Real, Vector, Matrix}=ones(Float64, length(J)),
                     Jdagger::Vector=map(dagger, J),
                     tmp::DenseOperator=deepcopy(rho0),
                     kwargs...)
    op2rho0 = op2*rho0
    tout = Float64[0.]
    exp_values = Complex128[expect(op1, op2rho0)]
    function fout(t, rho)
        push!(tout, t)
        push!(exp_values, expect(op1, rho))
    end
    steadystate.master(H, J; rho0=op2rho0, eps=eps, h0=h0, fout=fout,
                       Gamma=Gamma, Jdagger=Jdagger, tmp=tmp, kwargs...)
    return tout, exp_values
end


function correlationspectrum(omega_samplepoints::Vector{Float64},
                H::Operator, J::Vector, op::Operator;
                eps::Float64=1e-4,
                rho_ss::DenseOperator=steadystate.master(H, J; eps=eps),
                kwargs...)
    domega = minimum(diff(omega_samplepoints))
    dt = 2*pi/(omega_samplepoints[end] - omega_samplepoints[1])
    T = 2*pi/domega
    tspan = [0.:dt:T;]
    exp_values = correlation(tspan, rho_ss, H, J, dagger(op), op, kwargs...)
    # dtmin = minimum(diff(tspan))
    # T = tspan[end] - tspan[1]
    # domega = 2*pi/T
    # omega_min = -pi/dtmin
    # omega_max = pi/dtmin
    # omega_samplepoints = Float64[omega_min:domega:omega_max-domega/2;]
    S = Float64[]
    for omega=omega_samplepoints
        y = exp(1im*omega*tspan).*exp_values/pi
        I = 0im
        for j=1:length(tspan)-1
            I += (tspan[j+1] - tspan[j])*(y[j+1] + y[j])
        end
        I = I/2
        push!(S, real(I))
    end
    return omega_samplepoints, S
end


function correlationspectrum(H::Operator, J::Vector, op::Operator;
                eps::Float64=1e-4, h0=10.,
                rho_ss::DenseOperator=steadystate.master(H, J; eps=eps),
                kwargs...)
    tspan, exp_values = correlation(rho_ss, H, J, dagger(op), op, eps=eps, h0=h0, kwargs...)
    dtmin = minimum(diff(tspan))
    T = tspan[end] - tspan[1]
    tspan = Float64[0.:dtmin:T;]
    return correlationspectrum(tspan, H, J, op; eps=eps, rho_ss=rho_ss, kwargs...)
    # domega = 1./T
    # omega_min = -pi/dtmin
    # omega_max = pi/dtmin
    # omega_samplepoints = Float64[omega_min:domega:omega_max;]
    # S = Float64[]
    # for omega=omega_samplepoints
    #     y = exp(1im*omega*tspan).*exp_values/pi
    #     I = 0im
    #     for j=1:length(tspan)-1
    #         I += (tspan[j+1] - tspan[j])*(y[j+1] + y[j])
    #     end
    #     I = I/2
    #     push!(S, real(I))
    # end
    # return omega_samplepoints, S
end

end # module
