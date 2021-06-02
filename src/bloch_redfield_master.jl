"""
    bloch_redfield_tensor(H, a_ops; J=[], use_secular=true, secular_cutoff=0.1)

Create the super-operator for the Bloch-Redfield master equation such that ``\\dot ρ = R ρ`` based on the QuTiP implementation.

See QuTiP's documentation (http://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html) for more information and a brief derivation.


# Arguments
* `H`: Hamiltonian.
* `a_ops`: Nested list of [interaction operator, callback function] pairs for the Bloch-Redfield type processes where the callback function describes the environment spectrum for the corresponding interaction operator.
           The spectral functions must take the angular frequency as their only argument.
* `J=[]`: Vector containing the jump operators for the Linblad type processes (optional).
* `use_secular=true`: Specify whether or not to use the secular approximation.
* `secular_cutoff=0.1`: Cutoff to allow a degree of partial secularization. Terms are discarded if they are greater than (dw\\_min * secular cutoff) where dw\\_min is the smallest (non-zero) difference between any two eigenenergies of H.
                        This argument is only taken into account if use_secular=true.
"""
function bloch_redfield_tensor(H::AbstractOperator, a_ops; J=SparseOpType[], use_secular=true, secular_cutoff=0.1)

    # use the eigenbasis
    H_evals, transf_mat = eigen(DenseOperator(H).data)
    H_ekets = Ket{typeof(H.basis_l),Vector{eltype(transf_mat)}}[Ket(H.basis_l, transf_mat[:, i]) for i in 1:length(H_evals)]

    N = length(H_evals)
    K = length(a_ops)

    # Calculate Liouvillian for Lindblad terms (unitary part + dissipation from J (if given)):
    Uinv = inv(transf_mat)
    tmp = copy(Uinv)
    Heb = sparse(to_Heb(tmp, H, transf_mat, Uinv))
    L = liouvillian(Heb, [to_Heb(tmp, j, transf_mat, Uinv) for j ∈ J])

    #If only Lindblad collapse terms (no a_ops given)
    if K==0
        return L, H_ekets
    end

    #Transform interaction operators to Hamiltonian eigenbasis
    A = Array{eltype(transf_mat)}(undef, N, N, K)
    for k in 1:K
        A[:, :, k] = to_Heb(tmp, a_ops[k][1], transf_mat, Uinv).data
    end

    # Trasition frequencies between eigenstates
    W = transpose(H_evals) .- H_evals

    #Array for spectral functions evaluated at transition frequencies
    Jw = Array{eltype(transf_mat)}(undef, N, N, K)
    # Jw = zeros(Complex{Float64}, N, N, K)
    for k in 1:K
       # do explicit loops here
       for n in 1:N
           for m in 1:N
               Jw[m, n, k] = a_ops[k][2](W[n, m])
           end
       end
    end

    #Calculate secular cutoff scale
    W_flat = reshape(W, N*N)
    dw_min = minimum(abs.(W_flat[W_flat .!= 0.0]))

    #Pre-calculate mapping between global index I and system indices a,b
    Iabs = Array{Int}(undef, N*N, 3)
    indices = CartesianIndices((N,N))
    for I in 1:N*N
        Iabs[I, 1] = I
        Iabs[I, 2:3] = [indices[I].I...]
    end


    # ALTERNATIVE DENSE METHOD - Main Bloch-Redfield operators part
    data = zeros(eltype(Jw), N^2, N^2)
    Is = view(Iabs, :, 1)
    As = view(Iabs, :, 2)
    Bs = view(Iabs, :, 3)

    for (I, a, b) in zip(Is, As, Bs)

        if use_secular
            Jcds = zeros(Int, size(Iabs))
            for (row, (I2, a2, b2)) in enumerate(zip(Is, As, Bs))
                if abs.(W[a, b] - W[a2, b2]) < dw_min * secular_cutoff
                    Jcds[row, :] = [I2 a2 b2]
                end
            end
            Jcds = transpose(Jcds)
            Jcds = Jcds[Jcds .!= 0]
            Jcds = reshape(Jcds, 3, Int(length(Jcds)/3))
            Jcds = transpose(Jcds)

            Js = view(Jcds, :, 1)
            Cs = view(Jcds, :, 2)
            Ds = view(Jcds, :, 3)

        else
            Js = Is
            Cs = As
            Ds = Bs
        end


        for (J, c, d) in zip(Js, Cs, Ds)

            sum!(   view(data, I, J),   view(A, c, a, :) .* view(A, b, d, :) .* (view(Jw, c, a, :) + view(Jw, d, b, :) )  )
            # data[I, J] = 0.5 * sum(view(A, c, a, :) .* view(A, b, d, :) .* (view(Jw, c, a, :) + view(Jw, d, b, :) ))

            if b == d
                data[I, J] -= sum( view(A, a, :, :) .* view(A, c, :, :) .* view(Jw, c, :, :) )
            end

            if a == c
                data[I, J] -= sum( view(A, d, :, :) .* view(A, b, :, :) .* view(Jw, d, :, :) )
            end

            # data[I, J] *= 0.5

        end
    end

    data *= 0.5
    R = sparse(data) # Removes any zero values and converts to sparse array

    #Add Bloch-Redfield part to Linblad Liouvillian calculated earlier
    L.data = L.data + R

    return L, H_ekets

end #Function

#Define function for transforming to Hamiltonian eigenbasis
function to_Heb(tmp, op, U, Uinv)
    oper = sparse(op)
    mul!(tmp,op.data,U)
    mul!(oper.data,Uinv,tmp)
    return oper
end


"""
    timeevolution.master_bloch_redfield(tspan, rho0, R, H; <keyword arguments>)

Time-evolution according to a Bloch-Redfield master equation.


# Arguments
* `tspan`: Vector specifying the points of time for which output should
        be displayed.
* `rho0`: Initial density operator. Can also be a state vector which is
        automatically converted into a density operator.
* `H`: Arbitrary operator specifying the Hamiltonian.
* `R`: Bloch-Redfield tensor describing the time-evolution ``\\dot ρ = R ρ`` (see timeevolution.bloch\\_redfield\\_tensor).
* `fout=nothing`: If given, this function `fout(t, rho)` is called every time
        an output should be displayed. ATTENTION: The given state rho is not
        permanent! It is still in use by the ode solver and therefore must not
        be changed.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function master_bloch_redfield(tspan,
        rho0::Operator{B,B}, L::SuperOperator{Tuple{B,B},Tuple{B,B}},
        H::AbstractOperator{B,B}; fout::Union{Function,Nothing}=nothing,
        kwargs...) where {B}

    #Prep basis transf
    evals, transf_mat = eigen(dense(H).data)
    transf_op = DenseOperator(rho0.basis_l, transf_mat)
    inv_transf_op = DenseOperator(rho0.basis_l, inv(transf_mat))

    # rho as Ket and L as DataOperator
    basis_comp = rho0.basis_l^2
    rho0_eb = Ket(basis_comp, (inv_transf_op * rho0 * transf_op).data[:]) #Transform to H eb and convert to Ket
    L_ = isa(L, SparseSuperOpType) ? SparseOperator(basis_comp, L.data) : DenseOperator(basis_comp, L.data)

    # Derivative function
    dmaster_br_(t, rho, drho) = dmaster_br(drho, rho, L_)

    return integrate_br(tspan, dmaster_br_, rho0_eb, transf_op, inv_transf_op, fout; kwargs...)
end
master_bloch_redfield(tspan, psi::Ket, args...; kwargs...) = master_bloch_redfield(tspan, dm(psi), args...; kwargs...)

# Derivative ∂ₜρ = Lρ
function dmaster_br(drho, rho, L)
    QuantumOpticsBase.mul!(drho,L,rho)
end

# Integrate if there is no fout specified
function integrate_br(tspan, dmaster_br, rho,
                transf_op, inv_transf_op, ::Nothing;
                kwargs...)
    # Pre-allocate for in-place back-transformation from eigenbasis
    rho_out = copy(transf_op)
    tmp = copy(transf_op)
    tmp2 = copy(transf_op)

    # Define fout
    function fout(t, rho)
        tmp.data[:] = rho.data
        QuantumOpticsBase.mul!(tmp2,transf_op,tmp)
        QuantumOpticsBase.mul!(rho_out,tmp2,inv_transf_op)
        return copy(rho_out)
    end

    return integrate(tspan, dmaster_br, copy(rho.data), rho, copy(rho), fout; kwargs...)
end

# Integrate with given fout
function integrate_br(tspan, dmaster_br, rho,
                transf_op, inv_transf_op, fout::Function;
                kwargs...)
    # Pre-allocate for in-place back-transformation from eigenbasis
    rho_out = copy(transf_op)
    tmp = copy(transf_op)
    tmp2 = copy(transf_op)

    tspan_ = convert(Vector{float(eltype(tspan))}, tspan)

    # Perform back-transfomration before calling fout
    function fout_(t, rho)
        tmp.data[:] = rho.data
        QuantumOpticsBase.mul!(tmp2,transf_op,tmp)
        QuantumOpticsBase.mul!(rho_out,tmp2,inv_transf_op)
        return fout(t, rho_out)
    end

    return integrate(tspan_, dmaster_br, copy(rho.data), rho, copy(rho), fout_; kwargs...)
end
