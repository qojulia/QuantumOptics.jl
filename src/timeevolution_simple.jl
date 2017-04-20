module timeevolution_simple

using ..bases
using ..states
using ..operators
using ..operators_dense
using ODE


"""
Evaluate master equation for diagonal jump operators.
"""
function dmaster(rho::DenseOperator, H::Operator, gamma::Vector, J::Vector, Jdagger::Vector)
    drho = -1im * (H*rho - rho*H)
    for n = 1:length(J)
        drho = drho + gamma[n]*(J[n]*rho*Jdagger[n] - Jdagger[n]*(J[n]*rho)/Complex(2) - rho*Jdagger[n]*J[n]/Complex(2))
    end
    return drho
end

"""
Evaluate master equation for non-diagonal jump operators.
"""
function dmaster(rho::DenseOperator, H::Operator, gamma::Matrix, J::Vector, Jdagger::Vector)
    drho = -1im * (H*rho - rho*H)
    for m=1:length(J), n=1:length(J)
       drho += gamma[m,n]*(J[m]*rho*Jdagger[n] - Jdagger[n]*(J[m]*rho)/Complex(2) - rho*Jdagger[n]*J[m]/Complex(2))
    end
    return drho
end

"""
Integrate master equation.
"""
function master(T::Vector, rho0::DenseOperator, H::Operator, J::Vector;
                    Jdagger=map(dagger,J),
                    gamma::Union{Real, Vector, Matrix}=ones(Int, length(J)),
                    kwargs...)
    check_samebases(rho0, H)
    for j=J
        @assert typeof(j) <: Operator
        check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert typeof(j) <: Operator
        check_samebases(rho0, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(gamma)<:Real
        gamma = ones(typeof(gamma), length(J))*gamma
    end
    n = length(basis(rho0))
    N = n^2
    as_operator(x::Vector{Complex128}) = DenseOperator(rho0.basis_l, rho0.basis_r, reshape(x, n, n))
    as_vector(rho::DenseOperator) = reshape(rho.data, N)
    f(t::Float64, x::Vector{Complex128}) = as_vector(dmaster(as_operator(x), H, gamma, J, Jdagger))
    tout, x_t = ode45(f, as_vector(rho0), T; kwargs...)
    rho_t = DenseOperator[as_operator(x) for x=x_t]
    return tout, rho_t
end

master(T::Vector, psi0::Ket, H::Operator, J::Vector; kwargs...) = master(T, tensor(psi0, dagger(psi0)), H, J; kwargs...)


"""
Evaluate Schroedinger equation for ket states.
"""
function dschroedinger(psi::Ket, H::Operator)
    return -1im*H*psi
end

"""
Evaluate Schroedinger equation for bra states.
"""
function dschroedinger(psi::Bra, H::Operator)
    return 1im*psi*H
end

"""
Integrate Schroedinger equation.
"""
function schroedinger{T<:StateVector}(tspan::Vector, psi0::T, H::Operator; kwargs...)
    as_statevector(x::Vector{Complex128}) = T(psi0.basis, x)
    as_vector(psi::T) = psi.data
    f(t::Float64, x::Vector{Complex128}) = as_vector(dschroedinger(as_statevector(x), H))
    tout, x_t = ode45(f, as_vector(psi0), tspan; kwargs...)
    psi_t = T[as_statevector(x) for x=x_t]
    return tout, psi_t
end

end  # module
