module steadystate

using ..states, ..operators, ..operators_dense, ..superoperators
using ..timeevolution


"""
    steadystate.master(H, J; <keyword arguments>)

Calculate steady state using long time master equation evolution.

# Arguments
* `H`: Arbitrary operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary
        operator type.
* `rho0=dm(basisstate(b))`: Initial density operator. If not given the
        ``|0⟩⟨0|`` state in respect to the choosen basis is used.
* `tol=1e-3`: Tracedistance used as termination criterion.
* `hmin=1e-7`: Minimal time step used in the time evolution.
* `rates=ones(N)`: Vector or matrix specifying the coefficients for the
        jump operators.
* `Jdagger=dagger.(Jdagger)`: Vector containing the hermitian conjugates of the
        jump operators. If they are not given they are calculated automatically.
* `fout=nothing`: If given this function `fout(t, rho)` is called every time an
        output should be displayed. To limit copying to a minimum the given
        density operator `rho` is further used and therefore must not be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master(H::Operator, J::Vector;
                rho0::DenseOperator=tensor(basisstate(H.basis_l, 1), dagger(basisstate(H.basis_r, 1))),
                hmin=1e-7, tol=1e-3,
                rates::Union{Vector{Float64}, Matrix{Float64}, Void}=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Void}=nothing,
                kwargs...)
    t,u = timeevolution.master([0., Inf], rho0, H, J; rates=rates, Jdagger=Jdagger,
                        hmin=hmin, hmax=Inf,
                        display_initialvalue=false,
                        display_finalvalue=false,
                        display_intermediatesteps=true,
                        fout=fout,
                        steady_state = true,
                        tol = tol, kwargs...)
end

"""
    steadystate.eigenvector(L)
    steadystate.eigenvector(H, J)

Find steady state by calculating the eigenstate of the Liouvillian matrix `l`.
"""
function eigenvector(L::DenseSuperOperator)
    d, v = Base.eig(L.data)
    index = findmin(abs.(d))[2]
    data = reshape(v[:,index], length(L.basis_r[1]), length(L.basis_r[2]))
    op = DenseOperator(L.basis_r[1], L.basis_r[2], data)
    return op/trace(op)
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

eigenvector(H::Operator, J::Vector) = eigenvector(liouvillian(H, J))


end # module
