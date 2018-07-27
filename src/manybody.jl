module manybody

export ManyBodyBasis, fermionstates, bosonstates,
        manybodyoperator, onebodyexpect, occupation

import Base: ==
import ..states: basisstate
import ..fock: number, create, destroy
import ..nlevel: transition

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

"""
    ManyBodyBasis(b, occupations)

Basis for a many body system.

The basis has to know the associated one-body basis `b` and which occupation states
should be included. The occupations_hash is used to speed up checking if two
many-body bases are equal.
"""
mutable struct ManyBodyBasis <: Basis
    shape::Vector{Int}
    onebodybasis::Basis
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function ManyBodyBasis(onebodybasis::Basis, occupations::Vector{Vector{Int}})
        new([length(occupations)], onebodybasis, occupations, hash(hash.(occupations)))
    end
end

"""
    fermionstates(Nmodes, Nparticles)
    fermionstates(b, Nparticles)

Generate all fermionic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number.
"""
fermionstates(Nmodes::Int, Nparticles::Int) = _distribute_fermions(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[])
fermionstates(Nmodes::Int, Nparticles::Vector{Int}) = vcat([fermionstates(Nmodes, N) for N in Nparticles]...)
fermionstates(onebodybasis::Basis, Nparticles) = fermionstates(length(onebodybasis), Nparticles)

"""
    bosonstates(Nmodes, Nparticles)
    bosonstates(b, Nparticles)

Generate all bosonic occupation states for N-particles in M-modes.
`Nparticles` can be a vector to define a Hilbert space with variable
particle number.
"""
bosonstates(Nmodes::Int, Nparticles::Int) = _distribute_bosons(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[])
bosonstates(Nmodes::Int, Nparticles::Vector{Int}) = vcat([bosonstates(Nmodes, N) for N in Nparticles]...)
bosonstates(onebodybasis::Basis, Nparticles) = bosonstates(length(onebodybasis), Nparticles)

==(b1::ManyBodyBasis, b2::ManyBodyBasis) = b1.occupations_hash==b2.occupations_hash && b1.onebodybasis==b2.onebodybasis

"""
    basisstate(b::ManyBodyBasis, occupation::Vector{Int})

Return a ket state where the system is in the state specified by the given
occupation numbers.
"""
function basisstate(basis::ManyBodyBasis, occupation::Vector{Int})
    index = findfirst(basis.occupations, occupation)
    if index == 0
        throw(ArgumentError("Occupation not included in many-body basis."))
    end
    basisstate(basis, index)
end

function isnonzero(occ1, occ2, index)
    for i=1:length(occ1)
        if i == index
            if occ1[i] != occ2[i] + 1
                return false
            end
        else
            if occ1[i] != occ2[i]
                return false
            end
        end
    end
    true
end

"""
    create(b::ManyBodyBasis, index)

Creation operator for the i-th mode of the many-body basis `b`.
"""
function create(b::ManyBodyBasis, index::Int)
    result = SparseOperator(b)
    # <{m}_i| at |{m}_j>
    for i=1:length(b)
        occ_i = b.occupations[i]
        if occ_i[index] == 0
            continue
        end
        for j=1:length(b)
            if isnonzero(occ_i, b.occupations[j], index)
                result.data[i, j] = sqrt(occ_i[index])
            end
        end
    end
    result
end

"""
    destroy(b::ManyBodyBasis, index)

Annihilation operator for the i-th mode of the many-body basis `b`.
"""
function destroy(b::ManyBodyBasis, index::Int)
    result = SparseOperator(b)
    # <{m}_j| a |{m}_i>
    for i=1:length(b)
        occ_i = b.occupations[i]
        if occ_i[index] == 0
            continue
        end
        for j=1:length(b)
            if isnonzero(occ_i, b.occupations[j], index)
                result.data[j, i] = sqrt(occ_i[index])
            end
        end
    end
    result
end

"""
    number(b::ManyBodyBasis, index)

Particle number operator for the i-th mode of the many-body basis `b`.
"""
function number(b::ManyBodyBasis, index::Int)
    result = SparseOperator(b)
    for i=1:length(b)
        result.data[i, i] = b.occupations[i][index]
    end
    result
end

"""
    number(b::ManyBodyBasis)

Total particle number operator.
"""
function number(b::ManyBodyBasis)
    result = SparseOperator(b)
    for i=1:length(b)
        result.data[i, i] = sum(b.occupations[i])
    end
    result
end

function isnonzero(occ1, occ2, index1::Int, index2::Int)
    for i=1:length(occ1)
        if i == index1 && i == index2
            if occ1[i] != occ2[i]
                return false
            end
        elseif i == index1
            if occ1[i] != occ2[i] + 1
                return false
            end
        elseif i == index2
            if occ1[i] != occ2[i] - 1
                return false
            end
        else
            if occ1[i] != occ2[i]
                return false
            end
        end
    end
    true
end

"""
    transition(b::ManyBodyBasis, to::Int, from::Int)

Operator ``|\\mathrm{to}⟩⟨\\mathrm{from}|`` transferring particles between modes.
"""
function transition(b::ManyBodyBasis, to::Int, from::Int)
    result = SparseOperator(b)
    # <{m}_j| at_to a_from |{m}_i>
    for i=1:length(b)
        occ_i = b.occupations[i]
        if occ_i[from] == 0
            continue
        end
        for j=1:length(b)
            occ_j = b.occupations[j]
            if isnonzero(occ_j, occ_i, to, from)
                result.data[j, i] = sqrt(occ_i[from])*sqrt(occ_j[to])
            end
        end
    end
    result
end

# Calculate many-Body operator from one-body operator
"""
    manybodyoperator(b::ManyBodyBasis, op)

Create the many-body operator from the given one-body operator `op`.

The given operator can either be a one-body operator or a
two-body interaction. Higher order interactions are at the
moment not implemented.

The mathematical formalism for the one-body case is described by

```math
X = \\sum_{ij} a_i^† a_j ⟨u_i| x | u_j⟩
```

and for the interaction case by

```math
X = \\sum_{ijkl} a_i^† a_j^† a_k a_l ⟨u_i|⟨u_j| x |u_k⟩|u_l⟩
```

where ``X`` is the N-particle operator, ``x`` is the one-body operator and
``|u⟩`` are the one-body states associated to the
different modes of the N-particle basis.
"""
function manybodyoperator(basis::ManyBodyBasis, op::T)::T where T<:Operator
    @assert op.basis_l == op.basis_r
    if op.basis_l == basis.onebodybasis
        result =  manybodyoperator_1(basis, op)
    elseif op.basis_l == basis.onebodybasis ⊗ basis.onebodybasis
        result = manybodyoperator_2(basis, op)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis."))
    end
    result
end

function manybodyoperator_1(basis::ManyBodyBasis, op::DenseOperator)
    N = length(basis)
    S = length(basis.onebodybasis)
    result = DenseOperator(basis)
    @inbounds for n=1:N, m=1:N
        for j=1:S, i=1:S
            C = coefficient(basis.occupations[m], basis.occupations[n], [i], [j])
            if C != 0.
                result.data[m,n] += C*op.data[i,j]
            end
        end
    end
    return result
end

function manybodyoperator_1(basis::ManyBodyBasis, op::SparseOperator)
    N = length(basis)
    S = length(basis.onebodybasis)
    result = SparseOperator(basis)
    M = op.data
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            row = M.rowval[i]
            value = M.nzval[i]
            for m=1:N, n=1:N
                C = coefficient(basis.occupations[m], basis.occupations[n], [row], [colindex])
                if C != 0.
                    result.data[m, n] += C*value
                end
            end
        end
    end
    return result
end

function manybodyoperator_2(basis::ManyBodyBasis, op::DenseOperator)
    N = length(basis)
    S = length(basis.onebodybasis)
    @assert S^2 == length(op.basis_l)
    @assert S^2 == length(op.basis_r)
    result = DenseOperator(basis)
    op_data = reshape(op.data, S, S, S, S)
    occupations = basis.occupations
    @inbounds for m=1:N, n=1:N
        for l=1:S, k=1:S, j=1:S, i=1:S
            C = coefficient(occupations[m], occupations[n], [i, j], [k, l])
            result.data[m,n] += C*op_data[i, j, k, l]
        end
    end
    return result
end

function manybodyoperator_2(basis::ManyBodyBasis, op::SparseOperator)
    N = length(basis)
    S = length(basis.onebodybasis)
    result = SparseOperator(basis)
    occupations = basis.occupations
    rows = rowvals(op.data)
    values = nonzeros(op.data)
    @inbounds for column=1:S^2, j in nzrange(op.data, column)
        row = rows[j]
        value = values[j]
        for m=1:N, n=1:N
            # println("row:", row, " column:"column, ind_left)
            index = ind2sub((S, S, S, S), (column-1)*S^2 + row)
            C = coefficient(occupations[m], occupations[n], index[1:2], index[3:4])
            if C!=0.
                result.data[m,n] += C*value
            end
        end
    end
    return result
end


# Calculate expectation value of one-body operator
"""
    onebodyexpect(op, state)

Expectation value of the one-body operator `op` in respect to the many-body `state`.
"""
function onebodyexpect(op::Operator, state::Ket)
    @assert isa(state.basis, ManyBodyBasis)
    @assert op.basis_l == op.basis_r
    if state.basis.onebodybasis == op.basis_l
        result = onebodyexpect_1(op, state)
    # Not yet implemented:
    # elseif state.basis.basis ⊗ state.basis.basis == op.basis_l
    #     result = onebodyexpect_2(op, state)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis of the state."))
    end
    result
end

function onebodyexpect(op::Operator, state::Operator)
    @assert op.basis_l == op.basis_r
    @assert state.basis_l == state.basis_r
    @assert isa(state.basis_l, ManyBodyBasis)
    if state.basis_l.onebodybasis == op.basis_l
        result = onebodyexpect_1(op, state)
    # Not yet implemented
    # elseif state.basis.basis ⊗ state.basis.basis == op.basis_l
    #     result = onebodyexpect_2(op, state)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis of the state."))
    end
    result
end
onebodyexpect(op::Operator, states::Vector) = [onebodyexpect(op, state) for state=states]

function onebodyexpect_1(op::DenseOperator, state::Ket)
    N = length(state.basis)
    S = length(state.basis.onebodybasis)
    result = complex(0.)
    occupations = state.basis.occupations
    for m=1:N, n=1:N
        value = conj(state.data[m])*state.data[n]
        for i=1:S, j=1:S
            C = coefficient(occupations[m], occupations[n], [i], [j])
            if C != 0.
                result += C*op.data[i,j]*value
            end
        end
    end
    result
end

function onebodyexpect_1(op::DenseOperator, state::DenseOperator)
    N = length(state.basis_l)
    S = length(state.basis_l.onebodybasis)
    result = complex(0.)
    occupations = state.basis_l.occupations
    @inbounds for s=1:N, t=1:N
        value = state.data[t,s]
        for i=1:S, j=1:S
            C = coefficient(occupations[s], occupations[t], [i], [j])
            if C != 0.
                result += C*op.data[i,j]*value
            end
        end
    end
    result
end

function onebodyexpect_1(op::SparseOperator, state::Ket)
    N = length(state.basis)
    S = length(state.basis.onebodybasis)
    result = complex(0.)
    occupations = state.basis.occupations
    M = op.data
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            row = M.rowval[i]
            value = M.nzval[i]
            for m=1:N, n=1:N
                C = coefficient(occupations[m], occupations[n], [row], [colindex])
                if C != 0.
                    result += C*value*conj(state.data[m])*state.data[n]
                end
            end
        end
    end
    result
end

function onebodyexpect_1(op::SparseOperator, state::DenseOperator)
    N = length(state.basis_l)
    S = length(state.basis_l.onebodybasis)
    result = complex(0.)
    occupations = state.basis_l.occupations
    M = op.data
    @inbounds for colindex = 1:M.n
        for i=M.colptr[colindex]:M.colptr[colindex+1]-1
            row = M.rowval[i]
            value = M.nzval[i]
            for s=1:N, t=1:N
                C = coefficient(occupations[s], occupations[t], [row], [colindex])
                if C != 0.
                    result += C*value*state.data[t,s]
                end
            end
        end
    end
    result
end


"""
Calculate the matrix element <{m}|at_1...at_n a_1...a_n|{n}>.
"""
function coefficient(occ_m, occ_n, at_indices, a_indices)
    occ_m = copy(occ_m)
    occ_n = copy(occ_n)
    C = 1.
    for i=at_indices
        if occ_m[i] == 0
            return 0.
        end
        C *= sqrt(occ_m[i])
        occ_m[i] -= 1
    end
    for i=a_indices
        if occ_n[i] == 0
            return 0.
        end
        C *= sqrt(occ_n[i])
        occ_n[i] -= 1
    end
    if occ_m == occ_n
        return C
    else
        return 0.
    end
end

function _distribute_bosons(Nparticles::Int, Nmodes::Int, index::Int, occupations::Vector{Int}, results::Vector{Vector{Int}})
    if index==Nmodes
        occupations[index] = Nparticles
        push!(results, copy(occupations))
    else
        for n=Nparticles:-1:0
            occupations[index] = n
            _distribute_bosons(Nparticles-n, Nmodes, index+1, occupations, results)
        end
    end
    return results
end

function _distribute_fermions(Nparticles::Int, Nmodes::Int, index::Int, occupations::Vector{Int}, results::Vector{Vector{Int}})
    if (Nmodes-index)+1<Nparticles
        return results
    end
    if index==Nmodes
        occupations[index] = Nparticles
        push!(results, copy(occupations))
    else
        for n=min(1,Nparticles):-1:0
            occupations[index] = n
            _distribute_fermions(Nparticles-n, Nmodes, index+1, occupations, results)
        end
    end
    return results
end

end # module
