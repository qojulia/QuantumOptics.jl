module timeevolution_bloch_redfield_master

export bloch_redfield_tensor, bloch_redfield_dynamics, tidyup!

#using QuantumOptics, LinearAlgebra, SparseArrays, DifferentialEquations #, SubspaceTransportModel_v3
using ...bases, ...states, ...operators
using ...operators_dense, ...operators_sparse, ...superoperators
using LinearAlgebra, SparseArrays, OrdinaryDiffEq #DifferentialEquations


#Define an equivalent to QuTiP's 'Qobj.tidyup()' method
function tidyup!(obj::AbstractOperator, tol=1e-12)

    for ele in eachindex(obj.data)
        if abs(obj.data[ele]) < tol
            obj.data[ele] = complex(0, 0)
        end
    end
    return obj
end



#Define a vec2mat_index function
function vec2mat_index(N, I)
    j = Int(round(I/N, RoundDown))
    i = I - N*j
    return [i, j]
end



#Main BR tensor function
function bloch_redfield_tensor(H::AbstractOperator, a_ops::Array; c_ops=[], use_secular=false, secular_cutoff=0.1)

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

    #only Lindblad collapse terms
    if K==0
        Heb = to_Heb(H)
        L = liouvillian(Heb, to_Heb.(c_ops))
        return L, H_ekets
    end

    A = Array{Complex}(undef, N, N, K)
    for k in 1:K
        A[:, :, k] = to_Heb(a_ops[k][1]).data
    end
    Jw = Array{Complex}(undef, N, N, K)

    # pre-calculate matrix elements and spectral densities
    W = transpose(H_evals) .- H_evals


    for k in 1:K
       # do explicit loops here in case spectra_cb[k] can not deal with array arguments
       for n in 1:N
           for m in 1:N
               Jw[m, n, k] = a_ops[k][2](W[n, m])
           end
       end
    end

    #Calculate secular cutoff scale
    W_flat = reshape(W, N*N)
    dw_min = minimum(abs.(W_flat[W_flat .!= 0.0]))

    # pre-calculate mapping between global index I and system indices a,b
    Iabs = Array{Int}(undef, N*N, 3)
    for I in 1:N*N
        Iabs[I, 1] = I
        Iabs[I, 2:3] = vec2mat_index(N, I-1) .+ 1 #.+1 to account for 1 based indexing
    end

    # unitary part + dissipation from c_ops (if given):
    Heb = to_Heb(H)
    L = liouvillian(Heb, to_Heb.(c_ops))

    # dissipative part
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
    Another bug here was that sparse func is happy to create rectangular arrays which we dont want.
    Work around is to check if both R cols and R rows have a (N*N)th element and if not then add one with value zero.
    This ensures R is always and N*N x N*N matrix as expected.
    """
    if !any(rows .== N*N) || !any(cols .== N*N) #Check if row or column list has (N*N)th element, if not then add one
        push!(rows, N*N)
        push!(cols, N*N)
        push!(data, 0.0)
    end

    R = sparse(rows, cols, data) # BUG WAS THAT ROWS AND COLS WERE WRONG WAY ROUND IN THIS LINE...

    L.data = L.data + R

    return L, H_ekets

end #Function



#Function for obtaining dynamics from Bloch-Redfield tensor

function bloch_redfield_dynamics(tspan, step_spacing, init::Ket, L::SuperOperator, H::AbstractOperator) #Need H for basis transformations

    #Prep basis transf
    evals, transf_mat = eigen(dense(H).data)
    N = length(evals) #Hilbert space dimension

    #Define ODE problem
    ρ0 = tensor(init, dagger(init)).data
    ρ0_eb = reshape(inv(transf_mat) * ρ0 * transf_mat, N*N, 1) #Transform to H eb and convert to vector
    ρ_dot(ρ, p, t) = L.data * ρ
    prob = ODEProblem(ρ_dot, ρ0_eb, tspan)
    #Solve the ODE problem
    sol = solve(prob, saveat=step_spacing)

    #Convert all states back density matrix then back to site basis
    states = reshape.(sol.u, N, N)
    states = [transf_mat * st * inv(transf_mat) for st in states]
    states = [DenseOperator(H.basis_l, st) for st in states]

    return sol.t, states
end






end #Module
