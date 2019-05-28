module timeevolution_bloch_redfield_master

export bloch_redfield_tensor, master_bloch_redfield

import ..integrate

using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse, ...superoperators

using LinearAlgebra, SparseArrays


"""
    bloch_redfield_tensor(H, a_ops; c_ops=[], use_secular=true, secular_cutoff=0.1)
"""
function bloch_redfield_tensor(H::AbstractOperator, a_ops::Array; c_ops=[], use_secular=true, secular_cutoff=0.1)

    # use the eigenbasis
    H_evals, transf_mat = eigen(DenseOperator(H).data)
    H_ekets = [Ket(H.basis_l, transf_mat[:, i]) for i in 1:length(H_evals)]

    #Define function for transforming to Hamiltonian eigenbasis
    function to_Heb(op)
        #Copy oper
        oper = copy(op)
        #Transform underlying array
        oper.data = inv(transf_mat) * oper.data * transf_mat
        return oper
    end

    N = length(H_evals)
    K = length(a_ops)

    #If only Lindblad collapse terms
    if K==0
        Heb = to_Heb(H)
        L = liouvillian(Heb, to_Heb.(c_ops))
        return L, H_ekets
    end

    #Transform interaction operators to Hamiltonian eigenbasis
    A = Array{Complex}(undef, N, N, K)
    for k in 1:K
        A[:, :, k] = to_Heb(a_ops[k][1]).data
    end

    # Trasition frequencies between eigenstates
    W = transpose(H_evals) .- H_evals

    #Array for spectral functions evaluated at transition frequencies
    Jw = Array{Complex}(undef, N, N, K)
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
    indices = CartesianIndices((N,2))
    for I in 1:N*N
        Iabs[I, 1] = I
        Iabs[I, 2:3] = [indices[I].I...]
    end

    # Calculate Liouvillian for Lindblad temrs (unitary part + dissipation from c_ops (if given)):
    Heb = to_Heb(H)
    L = liouvillian(Heb, to_Heb.(c_ops))

    # Main Bloch-Redfield operators part
    rows = Int[]
    cols = Int[]
    data = Complex[]
    Is = view(Iabs, :, 1)
    As = view(Iabs, :, 2)
    Bs = view(Iabs, :, 3)

    for (I, a, b) in zip(Is, As, Bs)

        if use_secular
            Jcds = Int.(zeros(size(Iabs)))
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

            elem = 0.5 * sum(view(A, c, a, :) .* view(A, b, d, :) .* (view(Jw, c, a, :) + view(Jw, d, b, :) ))

            if b == d
            elem -= 0.5 * sum(transpose(view(A, a, :, :)) .* transpose(view(A, c, :, :)) .* transpose(view(Jw, c, :, :) ))
            end

            if a == c
            elem -= 0.5 * sum(transpose(view(A, d, :, :)) .* transpose(view(A, b, :, :)) .* transpose(view(Jw, d, :, :) ))
            end

            #Add element
            if abs(elem) != 0.0
                push!(rows, I)
                push!(cols, J)
                push!(data, elem)
            end
        end
    end


    """
    Need to be careful here since sparse function is happy to create rectangular arrays but we dont want this behaviour so need to make square explicitly.
    """
    if !any(rows .== N*N) || !any(cols .== N*N) #Check if row or column list has (N*N)th element, if not then add one
        push!(rows, N*N)
        push!(cols, N*N)
        push!(data, 0.0)
    end

    R = sparse(rows, cols, data) #Careful with rows/cols ordering...

    #Add Bloch-Redfield part to Linblad Liouvillian calculated earlier
    L.data = L.data + R

    return L, H_ekets

end #Function



#Function for obtaining dynamics from Bloch-Redfield tensor (Liouvillian)
function master_bloch_redfield(tspan::Vector{Float64},
        rho0::T, L::SuperOperator{Tuple{B,B},Tuple{B,B}},
        H::AbstractOperator{B,B}; fout::Union{Function,Nothing}=nothing,
        kwargs...) where {B<:Basis,T<:DenseOperator{B,B}}

    #Prep basis transf
    evals, transf_mat = eigen(dense(H).data)
    inv_transf_mat = inv(transf_mat)
    N = length(evals) #Hilbert space dimension

    # rho as Ket and L as DataOperator
    basis_comp = rho0.basis_l^2
    rho0_eb = Ket(basis_comp, (inv_transf_mat * rho0.data * transf_mat)[:]) #Transform to H eb and convert to vector
    drho = copy(rho0_eb)
    L_ = isa(L, DenseSuperOperator) ? DenseOperator(basis_comp, L.data) : SparseOperator(basis_comp, L.data)
    dmaster_br_(t::Float64, rho::T2, drho::T2) where T2<:Ket = dmaster_br(drho, rho, L_)

    # Define fout
    rho_out = copy(rho0)
    if isa(fout, Nothing)
        fout_(t::Float64, rho::T) = copy(rho)
    else
        fout_ = fout
    end
    # TODO: Make fout more efficient
    function _fout_(t::Float64, rho::Ket)
        rho_ = DenseOperator(rho.basis.bases[1], rho.basis.bases[1], transf_mat * reshape(rho.data,N,N) * inv_transf_mat)
        return fout_(t::Float64, rho_)
    end

    return integrate(tspan, dmaster_br_, copy(rho0_eb.data), rho0_eb, drho, _fout_; kwargs...)
end
master_bloch_redfield(tspan::Vector{Float64}, psi::Ket, args...) = master_bloch_redfield(tspan::Vector{Float64}, dm(psi), args...)

function dmaster_br(drho::T, rho::T, L::DataOperator{B,B}) where {B<:Basis,T<:Ket{B}}
    operators.gemv!(1.0, L, rho, 0.0, drho)
end


end #Module
