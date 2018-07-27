module printing

export set_printing

import Base: show

using ..bases, ..states
using ..operators, ..operators_dense, ..operators_sparse
using ..operators_lazytensor, ..operators_lazysum, ..operators_lazyproduct
using ..spin, ..fock, ..nlevel, ..particle, ..subspace, ..manybody, ..sparsematrix


"""
    QuantumOptics.set_printing(; standard_order, rounding_tol)

Set options for REPL output.

# Arguments
* `standard_order=false`: For performance reasons, the order of the tensor
    product is inverted, i.e. `tensor(a, b)=kron(b, a)`. When changing this
    to `true`, the output shown in the REPL will exhibit the correct order.
* `rounding_tol=1e-17`: Tolerance for floating point errors shown in the output.
"""
function set_printing(; standard_order::Bool=_std_order, rounding_tol::Real=_round_tol)
    global _std_order = standard_order
    global _round_tol = rounding_tol
    global machineprecorder = Int32(round(-log10(_round_tol), digits=0))
    nothing
end
set_printing(standard_order=false, rounding_tol=1e-17)

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
    d = denominator(x.spinnumber)
    n = numerator(x.spinnumber)
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
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        showarray_stdord(stream, x.data, x.basis.shape, false, header=false)
    end
end

function show(stream::IO, x::Bra)
    write(stream, "Bra(dim=$(length(x.basis)))\n  basis: $(x.basis)\n")
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        showarray_stdord(stream, x.data, x.basis.shape, false, header=false)
    end
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
    if !_std_order
        Base.showarray(stream, x.data, false; header=false)
    else
        showarray_stdord(stream, x.data, x.basis_l.shape, x.basis_r.shape, false, header=false)
    end
end

function show(stream::IO, x::SparseOperator)
    showoperatorheader(stream, x)
    if nnz(x.data) == 0
        write(stream, "\n    []")
    else
        if !_std_order
            show(stream, x.data)
        else
            showsparsearray_stdord(stream, x.data, x.basis_l.shape, x.basis_r.shape)
        end
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

"""
    ind2Nary(m::Int, dims::Vector{Int})

The inverse of `Nary2ind`.

# Example
```
julia> dims = [2,2,3];

julia> for i in 1:prod(dims)
           println(i,": ", ind2Nary(i,dims))
       end
1: [0, 0, 0]
2: [0, 0, 1]
3: [0, 0, 2]
4: [0, 1, 0]
5: [0, 1, 1]
6: [0, 1, 2]
7: [1, 0, 0]
8: [1, 0, 1]
9: [1, 0, 2]
10: [1, 1, 0]
11: [1, 1, 1]
12: [1, 1, 2]
```
"""
function ind2Nary(m::Int, dims::Vector{Int})
    m = m - 1
    nq = length(dims)
    ar = zeros(Int, nq)
    product = prod(dims[2:end])
    for ith in 1:nq-1
        d = div(m, product)
        m = m - d * product
        product = div(product, dims[ith+1])
        ar[ith] = d
    end
    ar[end] = m
    return ar
end

"""
    Nary2ind(x, dims) -> index

Convert composite N-arys to index.

# Example
```
julia> dims = [2,2,3];

julia> Nary2ind([1,1,0], dims)
10

julia> for i in 1:prod(dims)
           println(i,": ", Nary2ind(ind2Nary(i,dims),dims), ": ", ind2Nary(i,dims))
       end
1: 1: [0, 0, 0]
2: 2: [0, 0, 1]
3: 3: [0, 0, 2]
4: 4: [0, 1, 0]
5: 5: [0, 1, 1]
6: 6: [0, 1, 2]
7: 7: [1, 0, 0]
8: 8: [1, 0, 1]
9: 9: [1, 0, 2]
10: 10: [1, 1, 0]
11: 11: [1, 1, 1]
12: 12: [1, 1, 2]
```
"""
function Nary2ind(x::Vector{Int}, dims::Vector{Int})
    tmp = 0
    @assert length(x) == length(dims)
    nterms = length(x)
    tp = prod(dims[2:end])
    for i in 1:nterms-1
        tmp += x[i] * tp
        tp = div(tp, dims[i+1])
    end
    tmp += x[end] + 1
end

"""
    mirror_world_index(idx, dims)

Convert index of standard order to that of inversed order.
This function is named after the book, '鏡の中の物理学' (Physics in the mirror),
written by Tomonaga Shin'ichirō (Japanese physicist).
"""
function mirror_world_index(idx::Int, dims::Vector{Int})
    return Nary2ind( reverse(ind2Nary(idx, dims)), reverse(dims)  )
end

# Following program is modified:
# julia/base/show.jl
# https://github.com/JuliaLang/julia/blob/5cd144ffa328fdd4cd9e983b616c7205f9bb4f51/base/show.jl
# and
# julia/base/sparse/sparsematrix.jl
# https://github.com/JuliaLang/julia/blob/78831902cc412e298c5fcc94dae1e4382c8e7cd0/base/sparse/sparsematrix.jl

# Copyright (c) 2009-2018: Jeff Bezanson, Stefan Karpinski, Viral B. Shah,
# and other contributors:
#
# https://github.com/JuliaLang/julia/contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
`alignment_std(X, ldims, rdims, rows, cols, cols_if_complete, cols_otherwise, sep)` returns the
alignment for specified parts of array `X`, returning the (left,right) info.
It will look in X's `rows`, `cols` (both lists of indices)
and figure out what's needed to be fully aligned, for example looking all
the way down a column and finding out the maximum size of each element.
Parameter `sep::Integer` is number of spaces to put between elements.
`cols_if_complete` and `cols_otherwise` indicate screen width to use.
Alignment is reported as a vector of (left,right) tuples, one for each
column going across the screen.
"""
function alignment_std(io::IO, X::AbstractVecOrMat, ldims::Vector, rdims::Vector,
        rows::AbstractVector, cols::AbstractVector,
        cols_if_complete::Integer, cols_otherwise::Integer, sep::Integer)
    a = Tuple{Int, Int}[]
    for j in cols # need to go down each column one at a time
        l = r = 0
        for i in rows # plumb down and see what largest element sizes are
            aij = Base.alignment(io, X[mirror_world_index(i, ldims), mirror_world_index(j, rdims)])
            l = max(l, aij[1]) # left characters
            r = max(r, aij[2]) # right characters
        end
        push!(a, (l, r)) # one tuple per column of X, pruned to screen width
        if length(a) > 1 && sum(map(sum,a)) + sep*length(a) >= cols_if_complete
            pop!(a) # remove this latest tuple if we're already beyond screen width
            break
        end
    end
    if 1 < length(a) < length(indices(X,2))
        while sum(map(sum,a)) + sep*length(a) >= cols_otherwise
            pop!(a)
        end
    end
    return a
end

"""
`print_matrix_row_std(io, X, A, ldims, rdims, i, cols, sep)` produces the aligned output for
a single matrix row X[i, cols] where the desired list of columns is given.
The corresponding alignment A is used, and the separation between elements
is specified as string sep.
`print_matrix_row_std` will also respect compact output for elements.
"""
function print_matrix_row_std(io::IO,
        X::AbstractVecOrMat, A::Vector, ldims::Vector, rdims::Vector,
        i::Integer, cols::AbstractVector, sep::AbstractString)
    isempty(A) || first(indices(cols,1)) == 1 || throw(DimensionMismatch("indices of cols ($(indices(cols,1))) must start at 1"))
    for k = 1:length(A)
        j = cols[k]
        x = X[mirror_world_index(i, ldims), mirror_world_index(j, rdims)]
        a = Base.alignment(io, x)
        sx = sprint(0, show, x, env=io)
        l = repeat(" ", A[k][1]-a[1]) # pad on left and right as needed
        r = repeat(" ", A[k][2]-a[2])
        prettysx = Base.replace_in_print_matrix(X,mirror_world_index(i, ldims),mirror_world_index(j, rdims),sx)
        print(io, l, prettysx, r)
        if k < length(A); print(io, sep); end
    end
end


function print_matrix_std(io::IO, X::AbstractVecOrMat, ldims::Vector, rdims::Vector,
                      pre::AbstractString = " ",  # pre-matrix string
                      sep::AbstractString = "  ", # separator between elements
                      post::AbstractString = "",  # post-matrix string
                      hdots::AbstractString = "  \u2026  ",
                      vdots::AbstractString = "\u22ee",
                      ddots::AbstractString = "  \u22f1  ",
                      hmod::Integer = 5, vmod::Integer = 5)
    if !get(io, :limit, false)
        screenheight = screenwidth = typemax(Int)
    else
        sz = displaysize(io)
        screenheight, screenwidth = sz[1] - 4, sz[2]
    end
    screenwidth -= length(pre) + length(post)
    presp = repeat(" ", length(pre))  # indent each row to match pre string
    postsp = ""
    @assert strwidth(hdots) == strwidth(ddots)
    sepsize = length(sep)
    rowsA, colsA = indices(X,1), indices(X,2)
    m, n = length(rowsA), length(colsA)
    # To figure out alignments, only need to look at as many rows as could
    # fit down screen. If screen has at least as many rows as A, look at A.
    # If not, then we only need to look at the first and last chunks of A,
    # each half a screen height in size.
    halfheight = div(screenheight,2)
    if m > screenheight
        rowsA = [rowsA[1:halfheight]; rowsA[m-div(screenheight-1,2)+1:m]]
    end
    # Similarly for columns, only necessary to get alignments for as many
    # columns as could conceivably fit across the screen
    maxpossiblecols = div(screenwidth, 1+sepsize)
    if n > maxpossiblecols
        colsA = [colsA[1:maxpossiblecols]; colsA[(n-maxpossiblecols+1):n]]
    end
    A = alignment_std(io, X, ldims, rdims, rowsA, colsA, screenwidth, screenwidth, sepsize)
    # Nine-slicing is accomplished using print_matrix_row_std repeatedly
    if m <= screenheight # rows fit vertically on screen
        if n <= length(A) # rows and cols fit so just print whole matrix in one piece
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                print_matrix_row_std(io, X,A,ldims,rdims,i,colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != last(rowsA); println(io); end
            end
        else # rows fit down screen but cols don't, so need horizontal ellipsis
            c = div(screenwidth-length(hdots)+1,2)+1  # what goes to right of ellipsis
            Ralign = reverse(alignment_std(io, X, ldims, rdims, rowsA, reverse(colsA), c, c, sepsize)) # alignments for right
            c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)
            Lalign = alignment_std(io, X, ldims, rdims, rowsA, colsA, c, c, sepsize) # alignments for left of ellipsis
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                print_matrix_row_std(io, X,Lalign,ldims,rdims,i,colsA[1:length(Lalign)],sep)
                print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)))
                print_matrix_row_std(io, X,Ralign,ldims,rdims,i,n-length(Ralign)+colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != last(rowsA); println(io); end
            end
        end
    else # rows don't fit so will need vertical ellipsis
        if n <= length(A) # rows don't fit, cols do, so only vertical ellipsis
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                print_matrix_row_std(io, X,A,ldims, rdims,i,colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != rowsA[end]; println(io); end
                if i == rowsA[halfheight]
                    print(io, i == first(rowsA) ? pre : presp)
                    Base.print_matrix_vdots(io, vdots,A,sep,vmod,1)
                    println(io, i == last(rowsA) ? post : postsp)
                end
            end
        else # neither rows nor cols fit, so use all 3 kinds of dots
            c = div(screenwidth-length(hdots)+1,2)+1
            Ralign = reverse(alignment_std(io, X, ldims, rdims, rowsA, reverse(colsA), c, c, sepsize))
            c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)
            Lalign = alignment_std(io, X, ldims, rdims, rowsA, colsA, c, c, sepsize)
            r = mod((length(Ralign)-n+1),vmod) # where to put dots on right half
            for i in rowsA
                print(io, i == first(rowsA) ? pre : presp)
                print_matrix_row_std(io, X,Lalign,ldims, rdims,i,colsA[1:length(Lalign)],sep)
                print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)))
                print_matrix_row_std(io, X,Ralign,ldims, rdims,i,n-length(Ralign)+colsA,sep)
                print(io, i == last(rowsA) ? post : postsp)
                if i != rowsA[end]; println(io); end
                if i == rowsA[halfheight]
                    print(io, i == first(rowsA) ? pre : presp)
                    Base.print_matrix_vdots(io, vdots,Lalign,sep,vmod,1)
                    print(io, ddots)
                    Base.print_matrix_vdots(io, vdots,Ralign,sep,vmod,r)
                    println(io, i == last(rowsA) ? post : postsp)
                end
            end
        end
    end
end

function showarray_stdord(io::IO, X::AbstractVecOrMat, ldims::Vector, rdims::Vector, repr::Bool = true; header = true)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    if !isempty(X)
        punct = (" ", "  ", "")
        print_matrix_std(io, X, ldims, rdims, punct...)
    end
end
showarray_stdord(io::IO, X::Vector, dims::Vector, repr::Bool = true; header = true) = showarray_stdord(io, X, dims, [1], repr; header = header)


function showsparsearray_stdord(io::IO, S::SparseMatrixCSC, ldims::Vector, rdims::Vector)
    if get(io, :limit, false)
        rows = displaysize(io)[1]
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(max(S.m,S.n))
    sep = "\n  "
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    colval = zeros(Int, nnz(S))
    for col in 1:S.n, k in S.colptr[col] : (S.colptr[col+1]-1)
        colval[k] = col
    end
    rowval = S.rowval
    rowval_std = map(x -> mirror_world_index(x, reverse(ldims)), rowval)
    colval_std = map(x -> mirror_world_index(x, reverse(rdims)), colval)
    idx_nzval = map((x,y,z) -> (x,y,z), rowval_std, colval_std, S.nzval)
    sort!(idx_nzval, by = x -> (x[2], x[1]) )
    rowval_std = map(x -> x[1], idx_nzval)
    colval_std = map(x -> x[2], idx_nzval)
    nzval_std = map(x -> x[3], idx_nzval)

    for (k, val) in enumerate(nzval_std)
        if k < half_screen_rows || k > nnz(S)-half_screen_rows
            print(io, sep, '[', rpad(rowval_std[k], pad), ", ", lpad(colval_std[k], pad), "]  =  ")
            show(io, nzval_std[k])
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
    end
end

end # module
