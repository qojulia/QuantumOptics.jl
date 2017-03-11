module nparticles

export NParticleBasis, BosonicNParticleBasis, FermionicNParticleBasis, nparticleoperator, expect_firstquantization

import Base.==

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse


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

distribute_bosons(Nparticles::Int, Nmodes::Int) = _distribute_bosons(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[])

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

distribute_fermions(Nparticles::Int, Nmodes::Int) = _distribute_fermions(Nparticles, Nmodes, 1, zeros(Int, Nmodes), Vector{Int}[])


"""
Abstract basis for n-particle systems. (Second quantization)

The inheriting bosonic and fermionic bases all have the same fields:

Nparticles
    (Fixed) number of particles in the system.
Nmodes
    Number of modes in which the particles can be distributed.
occupations
    Basis states that specify how many particles are in which modes.

The only difference between Fermionic and Bosonic basis is which basis states
are included. These states are created automatically if the occupation
states are ommited when creating a FermionicNParticleBasis or a
BosonicNParticleBasis.
"""
abstract NParticleBasis <: Basis

function check_NParticleBasis_arguments(Nparticles::Int, Nmodes::Int, occupations::Vector{Vector{Int}})
    Nparticles < 0 && throw(ArgumentError("Can't have less than zero particles."))
    for occupation=occupations
        Nmodes != length(occupation) && throw(ArgumentError("Dimension of single particle basis has to be equal to the dimension of the N-particle basis vector."))
        any(occupation.<0) && throw(ArgumentError("Occupation numbers smaller than zero not possible."))
        sum(occupation) != Nparticles && throw(ArgumentError("Total occupation has to be equal to the particle number."))
    end
end

type BosonicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    basis::Basis
    Nparticles::Int
    Nmodes::Int
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function BosonicNParticleBasis(basis::Basis, Nparticles::Int, occupations::Vector{Vector{Int}})
        Nmodes = length(basis)
        check_NParticleBasis_arguments(Nparticles, Nmodes, occupations)
        new([length(occupations)], basis, Nparticles, Nmodes, occupations, hash(occupations))
    end
end

type FermionicNParticleBasis <: NParticleBasis
    shape::Vector{Int}
    basis::Basis
    Nparticles::Int
    Nmodes::Int
    occupations::Vector{Vector{Int}}
    occupations_hash::UInt

    function FermionicNParticleBasis(basis::Basis, Nparticles::Int, occupations::Vector{Vector{Int}})
        Nmodes = length(basis)
        check_NParticleBasis_arguments(Nparticles, Nmodes, occupations)
        for occupation=occupations
            any(occupation.>1) && throw(ArgumentError("Occupation numbers greater than zero not possible for Fermions."))
        end
        new([length(occupations)], basis, Nparticles, Nmodes, occupations, hash(occupations))
    end
end

BosonicNParticleBasis(Nmodes::Int, Nparticles::Int, occupations::Vector{Vector{Int}}) = BosonicNParticleBasis(GenericBasis(Nmodes), Nparticles, occupations)
BosonicNParticleBasis(Nmodes::Int, Nparticles::Int) = BosonicNParticleBasis(GenericBasis(Nmodes), Nparticles, distribute_bosons(Nparticles, Nmodes))
BosonicNParticleBasis(basis::Basis, Nparticles::Int) = BosonicNParticleBasis(basis, Nparticles, distribute_bosons(Nparticles, length(basis)))

FermionicNParticleBasis(Nmodes::Int, Nparticles::Int, occupations::Vector{Vector{Int}}) = FermionicNParticleBasis(GenericBasis(Nmodes), Nparticles, occupations)
FermionicNParticleBasis(Nmodes::Int, Nparticles::Int) = FermionicNParticleBasis(GenericBasis(Nmodes), Nparticles, distribute_fermions(Nparticles, Nmodes))
FermionicNParticleBasis(basis::Basis, Nparticles::Int) = FermionicNParticleBasis(basis, Nparticles, distribute_fermions(Nparticles, length(basis)))

=={T<:NParticleBasis}(b1::T, b2::T) = (b1.Nparticles==b2.Nparticles && b1.Nmodes==b2.Nmodes && b1.occupations_hash==b2.occupations_hash)


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

basis
    NParticleBasis
op
    An operator represented in the single-particle functions associated to the
    modes of the N-particle basis. This means the dimension of this operator has to
    be equal to the Nmodes of the N-particle basis.
"""
function nparticleoperator_1(basis::NParticleBasis, op::DenseOperator)
    N = length(basis)
    S = basis.Nmodes
    @assert basis.basis == op.basis_l
    @assert basis.basis == op.basis_r
    result = DenseOperator(basis)
    for m=1:N, n=1:N
        for i=1:S, j=1:S
            C = coefficient(basis.occupations[m], basis.occupations[n], [i], [j])
            if C != 0.
                result.data[m,n] += C*op.data[i,j]
            end
        end
    end
    return result
end

function nparticleoperator_1(basis::NParticleBasis, op::SparseOperator)
    N = length(basis)
    S = basis.Nmodes
    @assert basis.basis == op.basis_l
    @assert basis.basis == op.basis_r
    result = SparseOperator(basis)
    M = op.data
    @inbounds for colindex = 1:M.n
        @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
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
    be equal to the Nmodes of the N-particle basis.
"""
function nparticleoperator_2(nparticlebasis::NParticleBasis, op::DenseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.Nmodes
    @assert S^2 == length(op.basis_l)
    @assert S^2 == length(op.basis_r)
    result = DenseOperator(nparticlebasis)
    op_data = reshape(op.data, S, S, S, S)
    occupations = nparticlebasis.occupations
    for m=1:N, n=1:N
        for i=1:S, j=1:S, k=1:S, l=1:S
            C = coefficient(occupations[m], occupations[n], [i, j], [k, l])
            result.data[m,n] += C*op_data[i, j, k, l]
        end
    end
    return result
end

function nparticleoperator_2(nparticlebasis::NParticleBasis, op::SparseOperator)
    N = length(nparticlebasis)
    S = nparticlebasis.Nmodes
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
            C = coefficient(occupations[m], occupations[n], index[1:2], index[3:4])
            if C!=0.
                result.data[m,n] += C*value
            end
        end
    end
    return result
end

"""
Create the n-particle operator from the given 1st quantized operator.

The given operator can either be a single particle operator or a
particle-particle interaction. Higher order interactions are at the
moment not implemented.

Arguments
---------

basis
    A n-particle basis
op
    Dense or sparse operator in first quantization.
"""
function nparticleoperator{T<:Operator}(basis::NParticleBasis, op::T)::T
    @assert op.basis_l == op.basis_r
    if op.basis_l == basis.basis
        result =  nparticleoperator_1(basis, op)
    elseif op.basis_l == basis.basis ⊗ basis.basis
        result = nparticleoperator_2(basis, op)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis."))
    end
    result
end


function expect_firstquantization_1(op::DenseOperator, state::Ket)
    N = length(state.basis)
    S = state.basis.Nmodes
    @assert isa(state.basis, NParticleBasis)
    @assert op.basis_l == state.basis.basis
    @assert op.basis_r == state.basis.basis
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

function expect_firstquantization_1(op::DenseOperator, state::DenseOperator)
    @assert state.basis_l == state.basis_r
    @assert isa(state.basis_l, NParticleBasis)
    @assert op.basis_l == state.basis_l.basis
    @assert op.basis_r == state.basis_l.basis
    N = length(state.basis_l)
    S = state.basis_l.Nmodes
    result = complex(0.)
    occupations = state.basis_l.occupations
    for s=1:N, t=1:N
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

function expect_firstquantization_1(op::SparseOperator, state::Ket)
    N = length(state.basis)
    S = state.basis.Nmodes
    @assert isa(state.basis, NParticleBasis)
    @assert op.basis_l == state.basis.basis
    @assert op.basis_r == state.basis.basis
    result = complex(0.)
    occupations = state.basis.occupations
    M = op.data
    @inbounds for colindex = 1:M.n
        @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
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

function expect_firstquantization_1(op::SparseOperator, state::DenseOperator)
    @assert state.basis_l == state.basis_r
    @assert isa(state.basis_l, NParticleBasis)
    @assert op.basis_l == state.basis_l.basis
    @assert op.basis_r == state.basis_l.basis
    N = length(state.basis_l)
    S = state.basis_l.Nmodes
    result = complex(0.)
    occupations = state.basis_l.occupations
    M = op.data
    @inbounds for colindex = 1:M.n
        @inbounds for i=M.colptr[colindex]:M.colptr[colindex+1]-1
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
Expectation value of the operator given in first quantization.

Arguments
---------

op
    Dense or sparse operator in first quantization.
state
    Ket-state in second quantization.
"""
function expect_firstquantization(op::Operator, state::Ket)
    @assert isa(state.basis, NParticleBasis)
    @assert op.basis_l == op.basis_r
    if state.basis.basis == op.basis_l
        result = expect_firstquantization_1(op, state)
    # Not yet implemented:
    # elseif state.basis.basis ⊗ state.basis.basis == op.basis_l
    #     result = expect_firstquantization_2(op, state)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis of the state."))
    end
    result
end

"""
Expectation value of the operator given in first quantization.

Arguments
---------

op
    Dense or sparse operator in first quantization.
state
    Density operator in second quantization.
"""
function expect_firstquantization(op::Operator, state::Operator)::Complex128
    @assert op.basis_l == op.basis_r
    @assert state.basis_l == state.basis_r
    @assert isa(state.basis_l, NParticleBasis)
    if state.basis_l.basis == op.basis_l
        result = expect_firstquantization_1(op, state)
    # Not yet implemented
    # elseif state.basis.basis ⊗ state.basis.basis == op.basis_l
    #     result = expect_firstquantization_2(op, state)
    else
        throw(ArgumentError("The basis of the given operator has to either be equal to b or b ⊗ b where b is the 1st quantization basis associated to the nparticle basis of the state."))
    end
    result
end

"""
Calculate the matrix element <{m}|at_1...at_n a_1...a_n|{n}>.
"""
function coefficient(occ_m, occ_n, at_indices, a_indices)
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

end