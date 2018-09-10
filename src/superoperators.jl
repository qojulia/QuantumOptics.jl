module superoperators

export SuperOperator, DenseSuperOperator, SparseSuperOperator,
        spre, spost, liouvillian, exp

import Base: ==, *, /, +, -
import ..bases
import SparseArrays: sparse

using ..bases, ..operators, ..operators_dense, ..operators_sparse
using SparseArrays


"""
Base class for all super operator classes.

Super operators are bijective mappings from operators given in one specific
basis to operators, possibly given in respect to another, different basis.
To embed super operators in an algebraic framework they are defined with a
left hand basis `basis_l` and a right hand basis `basis_r` where each of
them again consists of a left and right hand basis.
```math
A_{bl_1,bl_2} = S_{(bl_1,bl_2) ↔ (br_1,br_2)} B_{br_1,br_2}
\\\\
A_{br_1,br_2} = B_{bl_1,bl_2} S_{(bl_1,bl_2) ↔ (br_1,br_2)}
```
"""
abstract type SuperOperator end

"""
    DenseSuperOperator(b1[, b2, data])

SuperOperator stored as dense matrix.
"""
mutable struct DenseSuperOperator <: SuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::Matrix{ComplexF64}
    function DenseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::Matrix{ComplexF64})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

function DenseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis})
    Nl = length(basis_l[1])*length(basis_l[2])
    Nr = length(basis_r[1])*length(basis_r[2])
    data = zeros(ComplexF64, Nl, Nr)
    DenseSuperOperator(basis_l, basis_r, data)
end


"""
    SparseSuperOperator(b1[, b2, data])

SuperOperator stored as sparse matrix.
"""
mutable struct SparseSuperOperator <: SuperOperator
    basis_l::Tuple{Basis, Basis}
    basis_r::Tuple{Basis, Basis}
    data::SparseMatrixCSC{ComplexF64, Int}
    function SparseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis}, data::SparseMatrixCSC{ComplexF64, Int})
        if length(basis_l[1])*length(basis_l[2]) != size(data, 1) ||
           length(basis_r[1])*length(basis_r[2]) != size(data, 2)
            throw(DimensionMismatch())
        end
        new(basis_l, basis_r, data)
    end
end

function SparseSuperOperator(basis_l::Tuple{Basis, Basis}, basis_r::Tuple{Basis, Basis})
    Nl = length(basis_l[1])*length(basis_l[2])
    Nr = length(basis_r[1])*length(basis_r[2])
    data = spzeros(ComplexF64, Nl, Nr)
    SparseSuperOperator(basis_l, basis_r, data)
end

SuperOperator(basis_l, basis_r, data::SparseMatrixCSC{ComplexF64, Int}) = SparseSuperOperator(basis_l, basis_r, data)
SuperOperator(basis_l, basis_r, data::Matrix{ComplexF64}) = DenseSuperOperator(basis_l, basis_r, data)


Base.copy(a::T) where {T<:SuperOperator} = T(a.basis_l, a.basis_r, copy(a.data))

operators.dense(a::SparseSuperOperator) = DenseSuperOperator(a.basis_l, a.basis_r, Matrix(a.data))
operators.dense(a::DenseSuperOperator) = copy(a)

sparse(a::DenseSuperOperator) = SparseSuperOperator(a.basis_l, a.basis_r, sparse(a.data))
sparse(a::SparseSuperOperator) = copy(a)

==(a::T, b::T) where {T<:SuperOperator} = samebases(a, b) && (a.data == b.data)

Base.length(a::SuperOperator) = length(a.basis_l[1])*length(a.basis_l[2])*length(a.basis_r[1])*length(a.basis_r[2])
bases.samebases(a::SuperOperator, b::SuperOperator) = samebases(a.basis_l[1], b.basis_l[1]) && samebases(a.basis_l[2], b.basis_l[2]) &&
                                                      samebases(a.basis_r[1], b.basis_r[1]) && samebases(a.basis_r[2], b.basis_r[2])
bases.multiplicable(a::SuperOperator, b::SuperOperator) = multiplicable(a.basis_r[1], b.basis_l[1]) && multiplicable(a.basis_r[2], b.basis_l[2])
bases.multiplicable(a::SuperOperator, b::AbstractOperator) = multiplicable(a.basis_r[1], b.basis_l) && multiplicable(a.basis_r[2], b.basis_r)


# Arithmetic operations
function *(a::SuperOperator, b::DenseOperator)
    check_multiplicable(a, b)
    data = a.data*reshape(b.data, length(b.data))
    return DenseOperator(a.basis_l[1], a.basis_l[2], reshape(data, length(a.basis_l[1]), length(a.basis_l[2])))
end
function *(a::SuperOperator, b::SparseOperator)
    check_multiplicable(a, b)
    data = a.data*reshape(b.data, length(b.data))
    return SparseOperator(a.basis_l[1], a.basis_l[2], reshape(data, length(a.basis_l[1]), length(a.basis_l[2])))
end

function *(a::SuperOperator, b::SuperOperator)
    check_multiplicable(a, b)
    return SuperOperator(a.basis_l, b.basis_r, a.data*b.data)
end

*(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data*b)
*(a::Number, b::SuperOperator) = SuperOperator(b.basis_l, b.basis_r, a*b.data)

/(a::SuperOperator, b::Number) = SuperOperator(a.basis_l, a.basis_r, a.data/b)

+(a::SuperOperator, b::SuperOperator) = (check_samebases(a, b); SuperOperator(a.basis_l, a.basis_r, a.data+b.data))

-(a::SuperOperator) = SuperOperator(a.basis_l, a.basis_r, -a.data)
-(a::SuperOperator, b::SuperOperator) = (check_samebases(a, b); SuperOperator(a.basis_l, a.basis_r, a.data-b.data))

"""
    spre(op)

Create a super-operator equivalent for right side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spre}(A) B = A B
```

holds. `op` can be a dense or a sparse operator.
"""
spre(op::AbstractOperator) = SuperOperator((op.basis_l, op.basis_l), (op.basis_r, op.basis_r), tensor(op, identityoperator(op)).data)

"""
Create a super-operator equivalent for left side operator multiplication.

For operators ``A``, ``B`` the relation

```math
    \\mathrm{spost}(A) B = B A
```

holds. `op` can be a dense or a sparse operator.
"""
spost(op::AbstractOperator) = SuperOperator((op.basis_r, op.basis_r), (op.basis_l, op.basis_l), kron(permutedims(op.data), identityoperator(op).data))


function _check_input(H::AbstractOperator, J::Vector, Jdagger::Vector, rates::Union{Vector{Float64}, Matrix{Float64}})
    for j=J
        @assert typeof(j) <: AbstractOperator
        check_samebases(H, j)
    end
    for j=Jdagger
        @assert typeof(j) <: AbstractOperator
        check_samebases(H, j)
    end
    @assert length(J)==length(Jdagger)
    if typeof(rates) == Matrix{Float64}
        @assert size(rates, 1) == size(rates, 2) == length(J)
    elseif typeof(rates) == Vector{Float64}
        @assert length(rates) == length(J)
    end
end


"""
    liouvillian(H, J; rates, Jdagger)

Create a super-operator equivalent to the master equation so that ``\\dot ρ = S ρ``.

The super-operator ``S`` is defined by

```math
S ρ = -\\frac{i}{ħ} [H, ρ] + \\sum_i J_i ρ J_i^† - \\frac{1}{2} J_i^† J_i ρ - \\frac{1}{2} ρ J_i^† J_i
```

# Arguments
* `H`: Hamiltonian.
* `J`: Vector containing the jump operators.
* `rates`: Vector or matrix specifying the coefficients for the jump operators.
* `Jdagger`: Vector containing the hermitian conjugates of the jump operators. If they
             are not given they are calculated automatically.
"""
function liouvillian(H::T, J::Vector{T};
            rates::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
            Jdagger::Vector{T}=dagger.(J)) where T<:AbstractOperator
    _check_input(H, J, Jdagger, rates)
    L = spre(-1im*H) + spost(1im*H)
    if typeof(rates) == Matrix{Float64}
        for i=1:length(J), j=1:length(J)
            jdagger_j = rates[i,j]/2*Jdagger[j]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i,j]*J[i]) * spost(Jdagger[j])
        end
    elseif typeof(rates) == Vector{Float64}
        for i=1:length(J)
            jdagger_j = rates[i]/2*Jdagger[i]*J[i]
            L -= spre(jdagger_j) + spost(jdagger_j)
            L += spre(rates[i]*J[i]) * spost(Jdagger[i])
        end
    end
    return L
end

"""
    exp(op::DenseSuperOperator)

Operator exponential which can for example used to calculate time evolutions.
"""
Base.exp(op::DenseSuperOperator) = DenseSuperOperator(op.basis_l, op.basis_r, exp(op.data))

end # module
