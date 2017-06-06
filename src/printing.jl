module printing

import Base: show

using Compat
using ..bases
using ..spin, ..fock, ..nlevel, ..particle, ..subspace, ..manybody
using ..states
using ..operators, ..operators_dense, ..operators_sparse
using ..operators_lazytensor, ..operators_lazysum, ..operators_lazyproduct

function show(stream::IO, x::GenericBasis)
    if length(x.shape) == 1
        write(stream, "Basis(dim=$(x.shape[1]))")
    else
        s = replace(string(x.shape), " ", "")
        write(stream, "Basis(shape=$s)")
    end
end

function show(stream::IO, x::CompositeBasis)
    write(stream, "[")
    for i in 1:length(x.bases)
        show(stream, x.bases[i])
        if i != length(x.bases)
            write(stream, " ⊗ ")
        end
    end
    write(stream, "]")
end

function show(stream::IO, x::SpinBasis)
    d = @compat denominator(x.spinnumber)
    n = @compat numerator(x.spinnumber)
    if d == 1
        write(stream, "Spin($n)")
    else
        write(stream, "Spin($n/$d)")
    end
end

function show(stream::IO, x::FockBasis)
    write(stream, "Fock(cutoff=$(x.N))")
end

function show(stream::IO, x::NLevelBasis)
    write(stream, "NLevel(N=$(x.N))")
end

function show(stream::IO, x::PositionBasis)
    write(stream, "Position(xmin=$(x.xmin), xmax=$(x.xmax), N=$(x.N))")
end

function show(stream::IO, x::MomentumBasis)
    write(stream, "Momentum(pmin=$(x.pmin), pmax=$(x.pmax), N=$(x.N))")
end

function show(stream::IO, x::SubspaceBasis)
    write(stream, "Subspace(superbasis=$(x.superbasis), states:$(length(x.basisstates)))")
end

function show(stream::IO, x::ManyBodyBasis)
    write(stream, "ManyBody(onebodybasis=$(x.onebodybasis), states:$(length(x.occupations)))")
end

function show(stream::IO, x::Ket)
    write(stream, "Ket(dim=$(length(x.basis)))\n  basis: $(x.basis)\n")
    Base.showarray(stream, x.data, false; header=false)
end

function show(stream::IO, x::Bra)
    write(stream, "Bra(dim=$(length(x.basis)))\n  basis: $(x.basis)\n")
    Base.showarray(stream, x.data, false; header=false)
end

function showoperatorheader(stream::IO, x::Operator)
    write(stream, "$(typeof(x).name.name)(dim=$(length(x.basis_l))x$(length(x.basis_r)))\n")
    if bases.samebases(x)
        write(stream, "  basis: ")
        show(stream, basis(x))
    else
        write(stream, "  basis left:  ")
        show(stream, x.basis_l)
        write(stream, "\n  basis right: ")
        show(stream, x.basis_r)
    end
end

show(stream::IO, x::Operator) = showoperatorheader(stream, x)

function show(stream::IO, x::DenseOperator)
    showoperatorheader(stream, x)
    write(stream, "\n")
    Base.showarray(stream, x.data, false; header=false)
end

function show(stream::IO, x::SparseOperator)
    showoperatorheader(stream, x)
    if nnz(x.data) == 0
        write(stream, "\n    []")
    else
        show(stream, x.data)
    end
end

function show(stream::IO, x::LazyTensor)
    showoperatorheader(stream, x)
    write(stream, "\n  operators: $(length(x.operators))")
    s = replace(string(x.indices), " ", "")
    write(stream, "\n  indices: $s")
end

function show(stream::IO, x::Union{LazySum, LazyProduct})
    showoperatorheader(stream, x)
    write(stream, "\n  operators: $(length(x.operators))")
end


end # module
