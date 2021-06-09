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
function master(H::AbstractOperator{B,B}, J::Vector;
                rho0::Operator{B,B}=tensor(basisstate(H.basis_l, 1), dagger(basisstate(H.basis_r, 1))),
                tol=1e-3,
                rates::Union{Vector, Matrix, Nothing}=nothing,
                Jdagger::Vector=dagger.(J),
                fout::Union{Function,Nothing}=nothing,
                kwargs...) where B<:Basis
    t,u = timeevolution.master([0., Inf], rho0, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout,
                        steady_state = true,
                        tol = tol, kwargs...)
end

"""
    steadystate.liouvillianspectrum(L)
    steadystate.liouvillianspectrum(H, J)

Calculate eigenspectrum of the Liouvillian matrix `L`. The eigenvalues and -states are
sorted according to the absolute value of the eigenvalues.

# Keyword arguments:
* `nev = min(10, length(L.basis_r[1])*length(L.basis_r[2]))`: Number of eigenvalues.
* `which = :LR`: Find eigenvalues with largest real part. Keyword for `eigs`
    function (ineffective for DenseSuperOperator).
* `kwargs...`:  Keyword arguments for the Julia `eigen` or `eigens` function.
"""
function liouvillianspectrum(L::DenseSuperOpType; nev::Int = min(10, length(L.basis_r[1])*length(L.basis_r[2])), which::Symbol = :LR, kwargs...)
    d, v = eigen(L.data; kwargs...)
    indices = sortperm(abs.(d))[1:nev]
    ops = DenseOpType[]
    for i in indices
        data = reshape(v[:,i], length(L.basis_r[1]), length(L.basis_r[2]))
        op = DenseOperator(L.basis_r[1], L.basis_r[2], data)
        push!(ops, op)
    end
    return d[indices], ops
end

function liouvillianspectrum(L::SparseSuperOpType; nev::Int = min(10, length(L.basis_r[1])*length(L.basis_r[2])), which::Symbol = :LR, kwargs...)
    d, v, nconv, niter, nmult, resid = try
        eigs(L.data; nev = nev, which = which, kwargs...)
    catch err
        if isa(err, SingularException) || isa(err, Arpack.XYEUPD_Exception) || isa(err, Arpack.XYAUPD_Exception)
            error("Arpack's eigs() algorithm failed; try using DenseOperators or change nev.")
        else
            rethrow(err)
        end
    end
    indices = sortperm(abs.(d))[1:nev]
    ops = DenseOpType[]
    for i in indices
        data = reshape(v[:,i], length(L.basis_r[1]), length(L.basis_r[2]))
        op = DenseOperator(L.basis_r[1], L.basis_r[2], data)
        push!(ops, op)
    end
    return d[indices], ops
end

liouvillianspectrum(H::AbstractOperator{B,B}, J::Vector; rates::Union{Vector, Matrix}=ones(Float64, length(J)), kwargs...) where B<:Basis = liouvillianspectrum(liouvillian(H, J; rates=rates); kwargs...)

"""
    steadystate.eigenvector(L)
    steadystate.eigenvector(H, J)

Find steady state by calculating the eigenstate with eigenvalue 0 of the Liouvillian matrix `L`, if it exists.

# Keyword arguments:
* `tol = 1e-9`: Check `abs(eigenvalue) < tol` to determine zero eigenvalue.
* `nev = 2`: Number of calculated eigenvalues. If `nev > 1` it is checked if there
    is only one eigenvalue with real part 0. No checks for `nev = 1`: use if
    faster or for avoiding convergence errors of `eigs`. Changing `nev` thus only
    makes sense when using SparseSuperOperator.
* `which = :LR`: Find eigenvalues with largest real part. Keyword for `eigs` function (ineffective for DenseSuperOperator).
* `kwargs...`:  Keyword arguments for the Julia `eigen` or `eigs` function.
"""
function eigenvector(L::SuperOperator; tol::Real = 1e-9, nev::Int = 2, which::Symbol = :LR, kwargs...)
    d, ops = liouvillianspectrum(L; nev = nev, which = which, kwargs...)
    if abs(d[1]) > tol
        error("Eigenvalue with smallest absolute value is not zero.")
    end
    if nev > 1
        if abs(real(d[2])) < tol
            @warn("Several eigenvalues with real part 0 detected; use steadystate.liouvillianspectrum to find out more.")
        end
    end
    return ops[1]/tr(ops[1])
end

eigenvector(H::AbstractOperator, J::Vector; rates::Union{Vector, Matrix}=ones(Float64, length(J)), kwargs...) = eigenvector(liouvillian(H, J; rates=rates); kwargs...)
