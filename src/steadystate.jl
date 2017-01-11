module steadystate

using ..states
using ..operators
using ..superoperators
using ..timeevolution
using ..metrics


type ConvergenceReached <: Exception end


"""
Calculate steady state using long time master equation evolution.

Arguments
---------

H
    Operator specifying the Hamiltonian.
J
    Vector of jump operators.

Keyword Arguments
-----------------

rho0
    Initial density operator. If not given the :math:`|0 \\rangle\\langle0|`
    state in respect to the choosen basis is used.
eps
    Tracedistance used as termination criterion.
hmin
    Minimal time step used in the time evolution.
Gamma
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
function master(H::Operator, J::Vector;
                rho0::DenseOperator=tensor(basis_ket(H.basis_l, 1), basis_bra(H.basis_r, 1)),
                eps::Float64=1e-3, hmin=1e-7,
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector=map(dagger, J),
                fout::Union{Function,Void}=nothing,
                tmp::DenseOperator=deepcopy(rho0),
                kwargs...)
    t0 = 0.
    rho0 = deepcopy(rho0)
    function fout_steady(t, rho)
        if fout!=nothing
            fout(t, rho)
        end
        dt = t - t0
        drho = metrics.tracedistance(rho0, rho)
        t0 = t
        rho0.data[:] = rho.data
        if drho/dt < eps
            throw(ConvergenceReached())
        end
    end
    try
        timeevolution.master([0., Inf], rho0, H, J; Gamma=Gamma, Jdagger=Jdagger,
                            hmin=hmin, hmax=Inf,
                            tmp=tmp,
                            display_initialvalue=false,
                            display_finalvalue=false,
                            display_intermediatesteps=true,
                            fout=fout_steady, kwargs...)
    catch e
        if !isa(e, ConvergenceReached)
            rethrow(e)
        end
    end
    return rho0
end

"""
Find steady state by calculating the eigenstate of the Liouvillian matrix.

Arguments
---------

L
    Dense or sparse super-operator.
"""
function eigenvector(L::DenseSuperOperator)
    d, v = Base.eig(L.data)
    index = findmin(abs(d))[2]
    data = reshape(v[:,index], length(L.basis_r[1]), length(L.basis_r[2]))
    return DenseOperator(L.basis_r[1], L.basis_r[2], data)
end

function eigenvector(L::SparseSuperOperator)
    d, v, nconv, niter, nmult, resid = try
      Base.eigs(L.data; nev=1, sigma=1e-30)
    catch err
      if isa(err, LinAlg.SingularException)
        error("Base.LinAlg.eigs() algorithm failed; try using DenseOperators")
      else
        rethrow(err)
      end
    end
    data = reshape(v[:,1], length(L.basis_r[1]), length(L.basis_r[2]))
    op = DenseOperator(L.basis_r[1], L.basis_r[2], data)
    return op/trace(op)
end

"""
Find steady state by calculating the eigenstate of the Liouvillian matrix.

Arguments
---------

H
    Operator specifying the Hamiltonian.
J
    Vector of jump operators.
"""
eigenvector(H::Operator, J::Vector) = eigenvector(liouvillian(H, J))


end # module
