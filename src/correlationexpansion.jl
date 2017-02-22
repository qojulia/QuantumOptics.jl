module correlationexpansion

import Base: trace, ==, +, -, *, /
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems,
                    gemv!, gemm!

using Combinatorics, Iterators
using ..bases
using ..operators
using ..operators_dense
using ..operators_lazy
using ..ode_dopri


typealias Mask BitArray{1}

function indices2mask(N::Int, indices::Vector{Int})
    m = Mask(N)
    for i in indices
        m[i] = true
    end
    m
end

function indices2mask(N::Int, indices::Vector{Int}, m::Mask)
    fill!(m, false)
    for i in indices
        m[i] = true
    end
    m
end

mask2indices(mask::Mask) = find(mask)

complement(N::Int, indices::Vector{Int}) = Int[i for i=1:N if i ∉ indices]

as_mask(N::Int, m::Mask) = (@assert length(m)==N; m)
function as_mask(N::Int, m)
    m = collect(m)
    operators.check_indices(N, m)
    indices2mask(N, m)
end
masks(N::Int) = Set(combinations(1:N))
masks(N::Int, order::Int) = Set(combinations(1:N, order))

subcorrelationmasks(mask::Mask) = [indices2mask(length(mask), indices) for indices in
        chain([combinations(mask2indices(mask), k) for k=2:sum(mask)-1]...)]


type CorrelationExpansion <: Operator
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::Vector{DenseOperator}
    masks::Vector{Vector{Mask}}
    correlations::Dict{Mask, DenseOperator}
end

function CorrelationExpansion(operators::Vector{DenseOperator},
            correlations::Dict{Mask, DenseOperator},
            factor::Number=1)
    N = length(operators)
    basis_l = CompositeBasis(Basis[op.basis_l for op in operators])
    basis_r = CompositeBasis(Basis[op.basis_r for op in operators])
    maxorder = 1
    for (mask, op) in correlations
        @assert length(mask) == N
        @assert sum(mask) > 1
        @assert op.basis_l == tensor(basis_l.bases[mask]...)
        @assert op.basis_r == tensor(basis_r.bases[mask]...)
        maxorder = max(sum(mask), maxorder)
    end
    masks = Vector{Mask}[Mask[] for i=1:maxorder]
    for (mask, op) in correlations
        push!(masks[sum(mask)], mask)
    end
    ops = [copy(op) for op in operators]
    cors = Dict{Mask, DenseOperator}(m=>deepcopy(op) for (m, op) in correlations)
    CorrelationExpansion(N, basis_l, basis_r, factor, ops, masks, cors)
end

function CorrelationExpansion(basis_l::CompositeBasis, basis_r::CompositeBasis, S)
    N = length(basis_l.bases)
    operators = [DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]
    correlations = Dict{Mask, DenseOperator}()
    for m in S
        m = as_mask(N, m)
        @assert sum(m) > 1
        correlations[m] = tensor(operators[m])
    end
    CorrelationExpansion(operators, correlations)
end
CorrelationExpansion(basis::CompositeBasis, S) = CorrelationExpansion(basis, basis, S)
function CorrelationExpansion(operators::Vector, S)
    N = length(operators)
    correlations = Dict{Mask, DenseOperator}()
    for op in operators
        @assert typeof(op) == DenseOperator
    end
    for m in S
        m = as_mask(N, m)
        @assert sum(m) > 1
        b_l = CompositeBasis([op.basis_l for op in operators[m]]...)
        b_r = CompositeBasis([op.basis_r for op in operators[m]]...)
        correlations[m] = DenseOperator(b_l, b_r)
    end
    CorrelationExpansion(operators, correlations)
end
CorrelationExpansion(operators::Vector) = CorrelationExpansion(operators, Dict{Mask, DenseOperator}())


"""
Tensor product of a correlation and the density operators of the other subsystems.

Arguments
---------
operators
    Tuple containing the reduced density operators of each subsystem.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.
correlation
    Correlation operator for the subsystems specified by the given mask.
"""
function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
            correlation::DenseOperator)
    imin = findfirst(mask)
    imax = findlast(mask)
    left_operators = operators[1:imin-1]
    right_operators = operators[imax+1:end]
    J = mask[imin:imax]
    if all(J)
        op = correlation
    else
        indices_cor = [i for i in imin:imax if mask[i]]
        indices_ops = [i for i in imin:imax if !mask[i]]
        perm = sortperm([indices_cor; indices_ops])
        mixed_ops = operators[[i for i in imin:imax if !mask[i]]]
        op = permutesystems(tensor(correlation, mixed_ops...), perm)
    end
    tensor(left_operators..., op, right_operators...)
end
function embedcorrelation(operators::Vector{DenseOperator}, mask::Mask,
            correlation::Number)
    @assert sum(mask) == 0
    tensor(operators...)*correlation
end
embedcorrelation(operators::Vector{DenseOperator}, indices::Vector{Int}, correlation) = embedcorrelation(operators, indices2mask(length(operators), indices), correlation)

"""
Calculate the normalized correlation of the subsystems specified by the given index mask.

Arguments
---------
rho
    Density operator of the total system.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.

Optional Arguments
------------------
operators
    A tuple containing the reduced density operators of the single subsystems.
subcorrelations
    A (mask->operator) dictionary storing already calculated correlations.
"""
function correlation(rho::DenseOperator, mask::Mask;
            operators::Vector{DenseOperator}=(N=length(mask); [ptrace(normalize(rho), complement(N, [i])) for i in 1:N]),
            subcorrelations::Dict{Mask, DenseOperator}=Dict{Mask, DenseOperator}())
    # Check if this correlation was already calculated.
    @assert length(mask) == length(rho.basis_l.bases)
    if mask in keys(subcorrelations)
        return subcorrelations[mask]
    end
    order = sum(mask)
    rho = normalize(rho)
    σ = ptrace(rho, mask2indices(!mask))
    σ -= tensor(operators[mask]...)
    for submask in subcorrelationmasks(mask)
        subcorrelation = correlation(rho, submask;
                                     operators=operators,
                                     subcorrelations=subcorrelations)
        σ -= embedcorrelation(operators[mask], submask[mask], subcorrelation)
    end
    subcorrelations[mask] = σ
    σ
end
correlation(rho::Operator, indices::Vector{Int}) = correlation(rho, indices2mask(length(rho.basis_l.bases), indices))
correlation(rho::CorrelationExpansion, mask::Mask) = rho.correlations[mask]

"""
Approximate a density operator by including only certain correlations.

Arguments
---------
rho
    The density operator that should be approximated.
masks
    A set containing an index mask for every correlation that should be
    included. A index mask is a tuple consisting of booleans which indicate
    if the n-th subsystem is included in the correlation.
"""
function approximate(rho::DenseOperator, masks)
    @assert typeof(rho.basis_l) == CompositeBasis
    N = length(rho.basis_l.bases)
    alpha = trace(rho)
    rho = normalize(rho)
    operators = [ptrace(rho, complement(N, [i])) for i in 1:N]
    subcorrelations = Dict{Mask, DenseOperator}() # Dictionary to store intermediate results
    correlations = Dict{Mask, DenseOperator}()
    for m in masks
        m = as_mask(N, m)
        correlations[m] = correlation(rho, m;
                                         operators=operators,
                                         subcorrelations=subcorrelations)
    end
    CorrelationExpansion(operators, correlations, alpha)
end
function approximate(rho::DenseOperator)
    @assert typeof(rho.basis_l) == CompositeBasis
    N = length(rho.basis_l.bases)
    approximate(rho, Set{Mask}())
end


ptrace(mask::Mask, indices::Vector{Int}) = mask[complement(length(mask), indices)]

function ptrace(rho::CorrelationExpansion, indices::Vector{Int})
    N = rho.N
    operators = rho.operators[complement(N, indices)]
    factors = [trace(op) for op in rho.operators]
    result = tensor(operators...)*prod(factors[indices])
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        factor = prod(factors[setdiff(indices, I)])
        if isempty(I ∩ indices)
            correlation = factor*rho.correlations[mask]
        else
            J = [i-sum(complement(N, I).<i) for i in I ∩ indices]
            correlation = factor*ptrace(rho.correlations[mask], J)
        end
        op = embedcorrelation(operators, ptrace(mask, indices), correlation)
        result += op
    end
    rho.factor*result
end

function Base.full(rho::CorrelationExpansion)
    result = tensor(rho.operators...)
    for (mask, correlation) in rho.correlations
        result += embedcorrelation(rho.operators, mask, correlation)
    end
    rho.factor*result
end

function *(op1::CorrelationExpansion, op2::LazyTensor)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = deepcopy(op1)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end
function *(op1::LazyTensor, op2::CorrelationExpansion)
    check_multiplicable(op1.basis_r, op2.basis_l)
    result = deepcopy(op2)
    gemm!(Complex(1.), op1, op2, Complex(0.), result)
    return result
end
*(a::Number, b::CorrelationExpansion) = CorrelationExpansion(b.operators, b.correlations, a*b.factor)
*(a::CorrelationExpansion, b::Number) = CorrelationExpansion(a.operators, a.correlations, a.factor*b)

function reduced_indices(I, I_)
    Int[findfirst(j->i==j, I) for i in I_]
end

function gemm!(alpha, a::LazyTensor, b::CorrelationExpansion, beta, result::CorrelationExpansion)
    N = b.N
    result.factor = a.factor*alpha*b.factor
    @assert beta == complex(0.)
    alpha = complex(1.)
    for i=1:N
        if i ∈ keys(a.operators)
            gemm!(alpha, a.operators[i], b.operators[i], beta, result.operators[i])
        else
            copy!(result.operators[i].data, b.operators[i].data)
        end
    end
    for mask in keys(b.correlations)
        I = mask2indices(mask)
        I_ = I ∩ keys(a.operators)
        op = result.correlations[mask]
        if isempty(I_)
            copy!(op.data, b.correlations[mask].data)
        else
            indices = reduced_indices(I, I_)
            operators = [a.operators[i] for i in I_]
            a_ = embed(op.basis_l, op.basis_r, indices, operators)
            gemm!(alpha, a_, b.correlations[mask], beta, op)
        end
    end
end

function gemm!(alpha, a::CorrelationExpansion, b::LazyTensor, beta, result::CorrelationExpansion)
    N = a.N
    result.factor = a.factor*alpha*b.factor
    @assert beta == complex(0.)
    alpha = complex(1.)
    for i=1:N
        if i ∈ keys(b.operators)
            gemm!(alpha, a.operators[i], b.operators[i], beta, result.operators[i])
        else
            copy!(result.operators[i].data, a.operators[i].data)
        end
    end
    for mask in keys(a.correlations)
        I = mask2indices(mask)
        I_ = I ∩ keys(b.operators)
        op = result.correlations[mask]
        if isempty(I_)
            copy!(op.data, a.correlations[mask].data)
        else
            indices = reduced_indices(I, I_)
            operators = [b.operators[i] for i in I_]
            b_ = embed(op.basis_l, op.basis_r, indices, operators)
            gemm!(alpha, a.correlations[mask], b_, beta, op)
        end
    end
end

type CachedPTrace
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    tmpmask::Mask
    operators::Vector{DenseOperator}
    operators_trace::Vector{Complex128}
    correlations::Dict{Mask, Dict{Mask, DenseOperator}}
    correlations_trace::Dict{Mask, Complex128}
end

function muladd(result::Matrix{Complex128}, factor::Complex128, op::Matrix{Complex128})
    for I in eachindex(result)
        result[I] += factor*op[I]
    end
end

function ptrace_into(op::CachedPTrace, I::Mask, indices::Vector{Int},
                    complement_indices::Vector{Int},
                    factor::Complex128, result::DenseOperator)
    N = op.N
    dims = prod(result.basis_l.shape)
    productfactor = factor
    for i in indices
        productfactor *= op.operators_trace[i]
    end
    ops = op.operators[complement_indices]
    for I_σ in keys(op.correlations)
        indices_σ = mask2indices(I_σ)
        complement_indices_σ = complement(N, indices_σ)
        factor_ = factor
        for i in indices
            if i ∉ indices_σ
                factor_ *= op.operators_trace[i]
            end
        end
        if all(I[indices_σ])
            productfactor += factor_*op.correlations_trace[I_σ]
        elseif all(I[complement_indices_σ])
            tmp = op.correlations[I_σ][!I[I_σ]].data
            muladd(result.data, factor_, tmp)
        else
            tmp = embedcorrelation(ops, I_σ[!I], op.correlations[I_σ][!I[I_σ]]).data
            muladd(result.data, factor_, tmp)
        end
    end
    tmp = tensor(ops...).data
    muladd(result.data, productfactor, tmp)
end

function cache_ptrace_into(rho::CorrelationExpansion, cache::CachedPTrace)
    N = rho.N
    correlations = cache.correlations
    correlations_trace = cache.correlations_trace
    cache.operators = rho.operators
    for i=1:N
        cache.operators_trace[i] = trace(rho.operators[i])
    end
    for (I_σ, σ) in rho.correlations
        dims = σ.basis_l.shape
        order = sum(I_σ)
        C = cache.correlations[I_σ]
        C[trues(order)] = σ
        for n=order-1:-1:1
            for indices in combinations(1:order, n)
                J = indices2mask(order, indices)
                i = complement(order, indices)[indmin(dims[!J])]
                J_sup = copy(J)
                J_sup[i] = true
                C[J] = ptrace(C[J_sup], sum(J_sup[1:i]))
            end
        end
        J_sup = falses(order)
        J_sup[indmin(dims)] = true
        correlations_trace[I_σ] = trace(C[J_sup])
    end
end

function add_into(op::CorrelationExpansion, result::CorrelationExpansion, cache::CachedPTrace)
    N = op.N
    cache_ptrace_into(op, cache)
    indices = [2:N;]
    mask = cache.tmpmask
    indices2mask(N, indices, mask)
    ptrace_into(cache, mask, indices, [1], op.factor, result.operators[1])
    for i=2:N
        indices[i-1] = i-1
        indices2mask(N, indices, mask)
        ptrace_into(cache, mask, indices, [i], op.factor, result.operators[i])
    end
    for (I_σ, σ) in op.correlations
        complement_indices = mask2indices(I_σ)
        indices = complement(N, complement_indices)
        indices2mask(N, indices, mask)
        ptrace_into(cache, mask, indices, complement_indices, op.factor, result.correlations[I_σ])
    end
end

function Base.fill!(rho::CorrelationExpansion, a::Number)
    for op in rho.operators
        fill!(op.data, a)
    end
    for σ in values(rho.correlations)
        fill!(σ.data, a)
    end
    rho.factor = 1.
end

function issubmask(submask::Mask, mask::Mask)
    if sum(submask) >= sum(mask)
        return false
    end
    for i=1:length(mask)
        if submask[i] && !mask[i]
            return false
        end
    end
    true
end


function _dmaster_J(rho::CorrelationExpansion,
                 Gamma::Vector{Float64},
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::CorrelationExpansion, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    for i=1:length(J)
        gemm!(Gamma[i], J[i], rho, complex(0.), tmp2)
        gemm!(complex(1.), tmp2, Jdagger[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(complex(-0.5), Jdagger[i], tmp2, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(-0.5*Gamma[i], rho, Jdagger[i], complex(0.), tmp2)
        gemm!(complex(1.), tmp2, J[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
end

function _dmaster_J(rho::CorrelationExpansion,
                 Gamma::Matrix{Float64},
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::CorrelationExpansion, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    for j=1:length(J), i=1:length(J)
        gemm!(Gamma[i,j], J[i], rho, complex(0.), tmp2)
        gemm!(complex(1.), tmp2, Jdagger[j], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(complex(-0.5), Jdagger[j], tmp2, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(-0.5*Gamma[i,j], rho, Jdagger[j], complex(0.), tmp2)
        gemm!(complex(1.), tmp2, J[i], complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
end

function dmaster(rho::CorrelationExpansion, H::LazySum,
                 Gamma,
                 J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor},
                 drho::CorrelationExpansion, tmp::Dict{String, Any})
    cache = tmp["cachedptrace"]
    tmp1 = tmp["tmp_rho1"]
    tmp2 = tmp["tmp_rho2"]
    fill!(drho, 0.)
    drho.factor = rho.factor
    for i in 1:length(H.operators)
        a = H.factors[i]
        h = H.operators[i]
        gemm!(-1im*a, h, rho, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
        gemm!(1im*a, rho, h, complex(0.), tmp1)
        add_into(tmp1, drho, cache)
    end
    _dmaster_J(rho, Gamma, J, Jdagger, drho, tmp)
    doperators = drho.operators
    dcorrelations = drho.correlations
    for order=2:length(rho.masks)
        for mask in rho.masks[order]
            I = mask2indices(mask)
            suboperators = rho.operators[I]
            # Tr{̇̇d/dt ρ}
            σ_I = dcorrelations[mask]
            # d/dt ρ^{s_k}
            for i = 1:order
                σ_I -= embedcorrelation(suboperators, [i], doperators[I[i]])
            end
            # d/dt σ^{s}
            for submask in keys(dcorrelations)
                if !issubmask(submask, mask)
                    continue
                end
                σ_I -= embedcorrelation(suboperators, submask[I], dcorrelations[submask])
                for i in setdiff(mask2indices(!submask), mask2indices(!mask))
                    ops = [i==j ? doperators[j] : rho.operators[j] for j in I]
                    σ_I -= embedcorrelation(ops, submask[I], rho.correlations[submask])
                end
            end
            dcorrelations[mask] = σ_I
        end
    end
end

function allocate_memory(rho0::CorrelationExpansion, H::LazySum, J::Vector{LazyTensor})
    D = Dict{String, Any}()
    D["tmp_rho1"] = deepcopy(rho0)
    D["tmp_rho2"] = deepcopy(rho0)
    correlations = Dict{Mask, Dict{Mask, DenseOperator}}()
    for mask in keys(rho0.correlations)
        correlations[mask] = Dict{Mask, DenseOperator}()
    end
    D["cachedptrace"] = CachedPTrace(rho0.N, rho0.basis_l, rho0.basis_r,
                                    Mask(rho0.N),
                                    rho0.operators,
                                    zeros(Complex128, rho0.N),
                                    correlations,
                                    Dict{Mask, Complex128}())
    D
end

function datalength(x::CorrelationExpansion)
    L = sum(Int[length(op.basis_l)*length(op.basis_r) for op in x.operators])
    L += sum(Int[length(op.basis_l)*length(op.basis_r) for op in values(x.correlations)])
    L
end

function as_vector(rho::CorrelationExpansion, x::Vector{Complex128})
    i = 0
    for op in rho.operators
        L_i = length(op.basis_l)*length(op.basis_r)
        data = vec(op.data)
        @inbounds x[i+1:i+L_i] = data
        i += L_i
    end
    for masks_n in rho.masks
        for mask in masks_n
            op = rho.correlations[mask]
            L_i = length(op.basis_l)*length(op.basis_r)
            data = vec(op.data)
            @inbounds x[i+1:i+L_i] = data
            i += L_i
        end
    end
    x
end

function as_operator(x::Vector{Complex128}, rho::CorrelationExpansion)
    i = 0
    for op in rho.operators
        L_i = length(op.basis_l)*length(op.basis_r)
        data = vec(op.data)
        @inbounds data[:] = x[i+1:i+L_i]
        i += L_i
    end
    for masks_n in rho.masks
        for mask in masks_n
            op = rho.correlations[mask]
            L_i = length(op.basis_l)*length(op.basis_r)
            data = vec(op.data)
            @inbounds data[:] = x[i+1:i+L_i]
            i += L_i
        end
    end
    rho
end

function integrate_master(dmaster::Function, tspan, rho0::CorrelationExpansion;
                fout::Union{Function,Void}=nothing, kwargs...)
    x0 = as_vector(rho0, zeros(Complex128, datalength(rho0)))
    f = (x->x)
    if fout==nothing
        tout = Float64[]
        xout = CorrelationExpansion[]
        function fout_(t, rho::CorrelationExpansion)
            push!(tout, t)
            push!(xout, deepcopy(rho))
        end
        f = fout_
    else
        f = fout
    end
    tmp = deepcopy(rho0)
    f_(t, x::Vector{Complex128}) = f(t, as_operator(x, tmp))
    ode(dmaster, float(tspan), x0, f_; kwargs...)
    return fout==nothing ? (tout, xout) : nothing
end

function master(tspan, rho0::CorrelationExpansion, H::LazySum, J::Vector{LazyTensor};
                Gamma::Union{Vector{Float64}, Matrix{Float64}}=ones(Float64, length(J)),
                Jdagger::Vector{LazyTensor} = LazyTensor[dagger(j) for j=J],
                fout::Union{Function,Void}=nothing,
                kwargs...)
    rho = deepcopy(rho0)
    drho = deepcopy(rho0)
    D = allocate_memory(rho0, H, J)
    function dmaster_(t, x::Vector{Complex128}, dx::Vector{Complex128})
        dmaster(as_operator(x, rho), H, Gamma, J, Jdagger, drho, D)
        as_vector(drho, dx)
    end
    integrate_master(dmaster_, tspan, rho0; fout=fout, kwargs...)
end

master(tspan, rho0, H::LazyTensor, J; kwargs...) = master(tspan, rho0, LazySum(H), J; kwargs...)



end # module