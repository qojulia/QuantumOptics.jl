module timeevolution_master

export master, master_nh, master_h

using ...bases
using ...states
using ...operators
using ...operators_dense
using ...ode_dopri

"""
Evaluate master equation for diagonal jump operators.
"""
function dmaster_h(rho::DenseOperator, H::Operator,
                    Gamma::Vector{Complex128}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for i=1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[i], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i], rho, Jdagger[i], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for non-diagonal jump operators.
"""
function dmaster_h(rho::DenseOperator, H::Operator,
                    Gamma::Matrix{Complex128}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(complex(0,-1.), H, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, H, complex(1.), drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)

        operators.gemm!(complex(-0.5), Jdagger[j], tmp, complex(1.), drho)

        operators.gemm!(Gamma[i,j], rho, Jdagger[j], complex(0.), tmp)
        operators.gemm!(complex(-0.5), tmp, J[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for non-hermitian Hamiltonian and diagonal jump operators.
"""
function dmaster_nh(rho::DenseOperator, Hnh::Operator, Hnh_dagger::Operator,
                    Gamma::Vector{Complex128}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for i=1:length(J)
        operators.gemm!(Gamma[i], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[i], complex(1.), drho)
    end
    return drho
end

"""
Evaluate master equation for non-hermitian Hamiltonian and non-diagonal jump operators.
"""
function dmaster_nh(rho::DenseOperator, Hnh::Operator, Hnh_dagger::Operator,
                    Gamma::Matrix{Complex128}, J::Vector, Jdagger::Vector,
                    drho::DenseOperator, tmp::DenseOperator)
    operators.gemm!(complex(0,-1.), Hnh, rho, complex(0.), drho)
    operators.gemm!(complex(0,1.), rho, Hnh_dagger, complex(1.), drho)
    for j=1:length(J), i=1:length(J)
        operators.gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp)
        operators.gemm!(complex(1.), tmp, Jdagger[j], complex(1.), drho)
    end
    return drho
end

"""
Integrate master equation with the given derivative function.

Arguments
---------

dmaster
    A function f(t, rho, drho) that calculates the time-derivative of rho at
    time t and stores the result in drho.
tspan
    Vector specifying the points of time for which output should be displayed.
rho0
    Initial density operator.


Keyword arguments
-----------------

fout (optional)
    If given this function fout(t, rho) is called every time an output should
    be displayed. To limit copying to a minimum the given density operator rho
    is further used and therefore must not be changed.

kwargs
    Further arguments are passed on to the ode solver.
"""
function integrate_master(dmaster::Function, tspan, rho0::DenseOperator; fout=nothing, kwargs...)
    nl = prod(rho0.basis_l.shape)
    nr = prod(rho0.basis_r.shape)
    N = nl*nr
    as_operator(x::Vector{Complex128}) = DenseOperator(rho0.basis_l, rho0.basis_r, reshape(x, nl, nr))
    as_vector(rho::DenseOperator) = reshape(rho.data, N)
    f = (x->x)
    if fout==nothing
        tout = Float64[]
        xout = DenseOperator[]
        function fout_(t, rho::DenseOperator)
            push!(tout, t)
            push!(xout, deepcopy(rho))
        end
        f = fout_
    else
        f = fout
    end
    f_(t, x::Vector{Complex128}) = f(t, as_operator(x))
    dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128}) = dmaster(t, as_operator(x), as_operator(dx))
    ode(dmaster_, float(tspan), as_vector(rho0), f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function _check_input(rho0::DenseOperator, H::Operator, J::Vector, Jdagger::Vector, Gamma::Union{Vector{Float64}, Matrix{Float64}})
    operators.check_samebases(rho0, H)
    for j=J
        @assert typeof(j) <: Operator
        operators.check_samebases(rho0, j)
    end
    for j=Jdagger
        @assert typeof(j) <: Operator
        operators.check_samebases(rho0, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(Gamma) == Matrix{Float64}
        @assert size(Gamma, 1) == size(Gamma, 2) == length(J)
    elseif typeof(Gamma) == Vector{Float64}
        @assert length(Gamma) == length(J)
    end
end

"""
Integrate the master equation with dmaster_h as derivative function.

For further information look at :func:`master(,::DenseOperator,::Operator,)`
"""
function master_h(tspan, rho0::DenseOperator, H::Operator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::DenseOperator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master_h(tspan, psi0::Ket, H::Operator, J::Vector; kwargs...) = master_h(tspan, tensor(psi0, dagger(psi0)), H, J; kwargs...)


"""
Integrate the master equation with dmaster_nh as derivative function.

In this case the given Hamiltonian is assumed to be the non-hermitian version.
For further information look at :func:`master(,::DenseOperator,::Operator,)`
"""
function master_nh(tspan, rho0::DenseOperator, Hnh::Operator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Hnhdagger::Operator=dagger(Hnh),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::DenseOperator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, Hnh, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master_nh(tspan, psi0::Ket, Hnh::Operator, J::Vector; kwargs...) = master_nh(tspan, tensor(psi0, dagger(psi0)), Hnh, J; kwargs...)


"""
Integrate the master equation with dmaster_nh as derivative function.

There are two implementations for integrating the master equation:

* ``master_h``: Usual formulation of the master equation.
* ``master_nh``: Variant with non-hermitian Hamiltonian.

The ``master`` function takes a normal Hamiltonian, calculates the
non-hermitian Hamiltonian and then calls master_nh which is slightly faster.

Arguments
---------

tspan
    Vector specifying the points of time for which output should be displayed.
rho0
    Initial density operator (must be a dense operator). Can also be a
    state vector which is automatically converted into a density operator.
H
    DenseOperator specifying the Hamiltonian.
J
    Vector containing all jump operators.


Keyword Arguments
-----------------

Gamma (optional)
    Vector or matrix specifying the coefficients for the jump operators.
Jdagger (optional)
    Vector containing the hermitian conjugates of the jump operators. If they
    are not given they are calculated automatically.
fout (optional)
    If given this function fout(t, rho) is called every time an output should
    be displayed. To limit copying to a minimum the given density operator rho
    is further used and therefore must not be changed.
kwargs
    Further arguments are passed on to the ode solver.
"""
function master(tspan, rho0::DenseOperator, H::Operator, J::Vector;
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::DenseOperator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Gamma = complex(Gamma)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_h(rho, H, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

function master(tspan, rho0::DenseOperator, H::DenseOperator, J::Vector{DenseOperator};
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector{DenseOperator}=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::DenseOperator=deepcopy(rho0),
                kwargs...)
    _check_input(rho0, H, J, Jdagger, Gamma)
    Hnh = deepcopy(H)
    if typeof(Gamma) == Matrix{Float64}
        for i=1:length(J), j=1:length(J)
            Hnh -= 0.5im*Gamma[i,j]*Jdagger[i]*J[j]
        end
    elseif typeof(Gamma) == Vector{Float64}
        for i=1:length(J)
            Hnh -= 0.5im*Gamma[i]*Jdagger[i]*J[i]
        end
    end
    Hnhdagger = dagger(Hnh)
    Gamma = complex(Gamma)
    dmaster_(t, rho::DenseOperator, drho::DenseOperator) = dmaster_nh(rho, Hnh, Hnhdagger, Gamma, J, Jdagger, drho, tmp)
    return integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master(tspan, psi0::Ket, H::Operator, J::Vector; kwargs...) = master(tspan, tensor(psi0, dagger(psi0)), H, J; kwargs...)

end #module
