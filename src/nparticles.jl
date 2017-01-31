module nparticles

export NParticleBasis, BosonicNParticleBasis, FermionicNParticleBasis, nparticleoperator_1, nparticleoperator_2

import Base.==

using ..bases, ..operators, ..operators_dense, ..operators_sparse


function distribute_bosons(particlenumber::Int, singleparticledimension::Int, index::Int=1, occupations::Vector{Int}=zeros(Int,singleparticledimension), results::Vector{Vector{Int}}=Vector{Int}[])
    if index==singleparticledimension
        occupations[index] = particlenumber
        push!(results, copy(occupations))
    else
        for n=particlenumber:-1:0
            occupations[index] = n
            distribute_bosons(particlenumber-n, singleparticledimension, index+1, occupations, results)
        end
    end
    return results
end

function distribute_fermions(particlenumber::Int, singleparticledimension::Int, index::Int=1, occupations::Vector{Int}=zeros(Int,singleparticledimension), results::Vector{Vector{Int}}=Vector{Int}[])
    if (singleparticledimension-index)+1<particlenumber
        return results
    end
    if index==singleparticledimension
        occupations[index] = particlenumber
        push!(results, copy(occupations))
    else
        for n=min(1,particlenumber):-1:0
            occupations[index] = n
            distribute_fermions(particlenumber-n, singleparticledimension, index+1, occupations, results)
        end
    end
    return results
end

"""
Abstract basis for n-particle systems. (Second quantization)

The inheriting bosonic and fermionic bases all have the same fields:

particlenumber
    (Fixed) number of particles in the system.
modenumber
    Number of modes in which the particles can be distributed.
occupations
    Basis states that specify how many particles are in which modes.

The only difference between Fermionic and Bosonic basis is which basis states
are included. These states are created automatically if the occupation
states are ommited when creating a FermionicNParticleBasis or a
BosonicNParticleBasis.
"""
abstract NParticleBasis <: Basis

function check_NParticleBasis_arguments(particlenumber::Int, modenumber::Int, occupations::Vector{Vector{Int}})
    particlenumber < 0 && throw(ArgumentError("Can't have less than zero particles."))
    for occupation=occupations
        modenumber != length(occupation) && throw(ArgumentError("Dimension of single particle basis has to be equal to the dimension of the N-particle basis vector."))
        any(occupation.<0) && throw(ArgumentError("Occupation numbers smaller than zero not possible."))
        sum(occupation) != particlenumber && throw(ArgumentError("Total occupation has to be equal to the particle number."))
    end
end

type BosonicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    particlenumber::Int
    modenumber::Int
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function BosonicNParticleBasis(particlenumber::Int, modenumber::Int, occupations::Vector{Vector{Int}})
        check_NParticleBasis_arguments(particlenumber, modenumber, occupations)
        new([length(occupations)], particlenumber, modenumber, occupations, hash(occupations))
    end
end

type FermionicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    particlenumber::Int
    modenumber::Int
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function FermionicNParticleBasis(particlenumber::Int, modenumber::Int, occupations::Vector{Vector{Int}})
        check_NParticleBasis_arguments(particlenumber, modenumber, occupations)
        for occupation=occupations
            any(occupation.>1) && throw(ArgumentError("Occupation numbers greater than zero not possible for Fermions."))
        end
        new([length(occupations)], particlenumber, modenumber, occupations, hash(occupations))
    end
end

BosonicNParticleBasis(particlenumber::Int, modenumber::Int) = BosonicNParticleBasis(particlenumber, modenumber, distribute_bosons(particlenumber, modenumber))
FermionicNParticleBasis(particlenumber::Int, modenumber::Int) = FermionicNParticleBasis(particlenumber, modenumber, distribute_fermions(particlenumber, modenumber))


=={T<:NParticleBasis}(b1::T, b2::T) = (b1.particlenumber==b2.particlenumber && b1.modenumber==b2.modenumber && b1.occupations_hash==b2.occupations_hash)

function nparticleoperator(nparticlebasis::NParticleBasis, op::DenseOperator; rank::Int=1)
    rank<1 && throw(ArgumentError("Rank has to be greater than zero."))
    rank>=2 && throw(ArgumentError("Not yet implemented for ranks greater than one."))
    result = Operator(nparticlebasis)
    N = length(nparticlebasis)
    S = nparticlebasis.modenumber
    for m=1:N, n=1:N
        if m==n
            for i=1:nparticlebasis.modenumber
                result.data[m,m] += op.data[i,i]*basis.occupations[m][i]
            end
        end
        indices = indices_adaggeri_aj(basis.occupations[m], basis.occupations[n])
        if !isnull(indices)
            i, j = get(indices)
            Ni = basis.occupations[n][i]
            Nj = basis.occupations[n][j]+1
            op.data[m,n] += result.data[i,j]*sqrt(Ni*Nj)
        end
    end
    return op
end

"""
Create a N-particle operator from a single particle operator.

The mathematical formalism is described by

.. math::

    X = \\sum_{ij} a_i^\\dagger a_j
                    \\left\\langle u_i \\right|
                    x
                    \\left| u_j \\right\\rangle

where :math:`X` is the N-particle operator, :math:`x` is the single particle operator and
:math:`\\left| u \\right\\rangle` are the single particle states associated to the
different modes of the N-particle basis.

Arguments
---------

nparticlebasis
    NParticleBasis
op
    An operator represented in the single-particle functions associated to the
    modes of the N-particle basis. This means the dimension of this operator has to
    be equal to the modenumber of the N-particle basis.
"""
function nparticleoperator_1(nparticlebasis::NParticleBasis, op::DenseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.modenumber
    @assert S == length(op.basis_l)
    @assert S == length(op.basis_r)
    result = DenseOperator(nparticlebasis)
    occupations = nparticlebasis.occupations
    for m=1:N, n=1:N
        for i=1:S, j=1:S
            C = coeff(occupations[m], occupations[n], [i], [j])
            if C!=0.
                result.data[m,n] += C*op.data[i,j]
            end
        end
    end
    return result
end

function nparticleoperator_1(nparticlebasis::NParticleBasis, op::SparseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.modenumber
    @assert S == length(op.basis_l)
    @assert S == length(op.basis_r)
    result = SparseOperator(nparticlebasis)
    occupations = nparticlebasis.occupations
    rows = rowvals(op.data)
    values = nonzeros(op.data)
    for column=1:S, j in nzrange(op.data, column)
        row = rows[j]
        value = values[j]
        for m=1:N, n=1:N
            C = coeff(occupations[m], occupations[n], [row], [column])
            if C!=0.
                result.data[m,n] += C*value
            end
        end
    end
    return result
end



"""
Create a N-particle operator from a two-particle operator.

The mathematical formalism is described by

.. math::

    X = \\sum_{ijkl} a_i^\\dagger a_j^\\dagger a_k a_l
            \\left\\langle u_i \\right| \\left\\langle u_j \\right|
            x
            \\left| u_k \\right\\rangle \\left| u_l \\right\\rangle

where :math:`X` is the N-particle operator, :math:`x` is the two-particle operator and
:math:`\\left| u \\right\\rangle` are the single particle states associated to the
different modes of the N-particle basis.

Arguments
---------

nparticlebasis
    NParticleBasis
op
    A two particle operator represented in the single particle functions associated to the
    modes of the N-particle basis. This means the dimension of this operator has to
    be equal to the modenumber of the N-particle basis.
"""
function nparticleoperator_2(nparticlebasis::NParticleBasis, op::DenseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.modenumber
    @assert S^2 == length(op.basis_l)
    @assert S^2 == length(op.basis_r)
    result = DenseOperator(nparticlebasis)
    op_data = reshape(op.data, S, S, S, S)
    occupations = nparticlebasis.occupations
    for m=1:N, n=1:N
        for i=1:S, j=1:S, k=1:S, l=1:S
            C = coeff(occupations[m], occupations[n], [i, j], [k, l])
            result.data[m,n] += C*op_data[i, j, k, l]
        end
    end
    return result
end

function nparticleoperator_2(nparticlebasis::NParticleBasis, op::SparseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.modenumber
    @assert S^2 == length(op.basis_l)
    @assert S^2 == length(op.basis_r)
    result = SparseOperator(nparticlebasis)
    occupations = nparticlebasis.occupations
    rows = rowvals(op.data)
    values = nonzeros(op.data)
    for column=1:S^2, j in nzrange(op.data, column)
        row = rows[j]
        value = values[j]
        for m=1:N, n=1:N
            # println("row:", row, " column:"column, ind_left)
            index = ind2sub((S, S, S, S), (column-1)*S^2 + row)
            C = coeff(occupations[m], occupations[n], index[1:2], index[3:4])
            if C!=0.
                result.data[m,n] += C*value
            end
        end
    end
    return result
end



"""
Calculate the matrix element <{m}|at_1...at_n a_1...a_n|{n}>.
"""
function coeff(occ_m, occ_n, at_indices, a_indices)
    occ_m = deepcopy(occ_m)
    occ_n = deepcopy(occ_n)
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

"""
Find indices i and j where <{m}|a^t_i a_j |{n}> is not zero.

Returns null if this condition is never fullfilled for the given states.
"""
function indices_adaggeri_aj(occ_m::Vector{Int}, occ_n::Vector{Int})
    null = Nullable{Tuple{Int,Int}}()
    idx_create = 0
    idx_destroy = 0
    for i=1:length(occ_m)
        if occ_m[i]==occ_n[i]
            continue
        end
        delta = occ_n[i] - occ_m[i]
        if delta==-1
            if idx_destroy != 0
                return null
            end
            idx_destroy = i
            N_destroy = occ_n[i]
        elseif delta==1
            if idx_create != 0
                return null
            end
            idx_create = i
            N_create = occ_n[i]+1
        else
            return null
        end
    end
    return Nullable((idx_create, idx_destroy))
end


"""
Find indices i and j where <{m}|a^t_i a_j |{n}> is not zero.

Returns null if this condition is never fullfilled for the given states.
"""
function indices_adaggeri_adaggerj_ak_al(occ_m::Vector{Int}, occ_n::Vector{Int})
    null = Nullable{Tuple{Int,Int,Int,Int}}()
    i = 0
    j = 0
    k = 0
    l = 0
    for i=1:length(occ_m)
        if occ_m[i]==occ_n[i]
            continue
        end
        delta = occ_n[i] - occ_m[i]
        if delta==-1
            if idx_destroy != 0
                return null
            end
            idx_destroy = i
            N_destroy = occ_n[i]
        elseif delta==1
            if idx_create != 0
                return null
            end
            idx_create = i
            N_create = occ_n[i]+1
        else
            return null
        end
    end
    return Nullable((idx_create, idx_destroy))
end

end