module fock

import Base.==

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export FockBasis, number, destroy, create, fockstate, coherentstate, qfunc, displace


"""
    FockBasis(N)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Note that the dimension of this basis then is N+1.
"""
type FockBasis <: Basis
    shape::Vector{Int}
    N::Int
    function FockBasis(N::Int)
        if N < 0
            throw(DimensionMismatch())
        end
        new([N+1], N)
    end
end


==(b1::FockBasis, b2::FockBasis) = b1.N==b2.N

"""
    number(b::FockBasis)

Number operator for the specified Fock space.
"""
function number(b::FockBasis)
    diag = Complex128[complex(x) for x=0:b.N]
    data = spdiagm(diag, 0, b.N+1, b.N+1)
    SparseOperator(b, data)
end

"""
    destroy(b::FockBasis)

Annihilation operator for the specified Fock space.
"""
function destroy(b::FockBasis)
    diag = Complex128[complex(sqrt(x)) for x=1:b.N]
    data = spdiagm(diag, 1, b.N+1, b.N+1)
    SparseOperator(b, data)
end

"""
    create(b::FockBasis)

Creation operator for the specified Fock space.
"""
function create(b::FockBasis)
    diag = Complex128[complex(sqrt(x)) for x=1:b.N]
    data = spdiagm(diag, -1, b.N+1, b.N+1)
    SparseOperator(b, data)
end

"""
    displace(b::FockBasis, alpha)

Displacement operator ``D(α)`` for the specified Fock space.
"""
displace(b::FockBasis, alpha::Number) = expm(full(alpha*create(b) - conj(alpha)*destroy(b)))

"""
    fockstate(b::FockBasis, n)

Fock state ``|n⟩`` for the specified Fock space.
"""
function fockstate(b::FockBasis, n::Int)
    @assert n <= b.N
    basisstate(b, n+1)
end

"""
    coherentstate(b::FockBasis, alpha)

Coherent state ``|α⟩`` for the specified Fock space.
"""
function coherentstate(b::FockBasis, alpha::Number, result=Ket(b, Vector{Complex128}(length(b))))
    alpha = complex(alpha)
    data = result.data
    data[1] = exp(-abs2(alpha)/2)
    @inbounds for n=1:b.N
        data[n+1] = data[n]*alpha/sqrt(n)
    end
    return result
end

"""
    qfunc(x, α)
    qfunc(x, xvec, yvec)

Husimi Q representation ``⟨α|ρ|α⟩/π`` for the given state or operator `x`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`.
"""
function qfunc(rho::Operator, alpha::Complex128,
                tmp1=Ket(basis(rho), Vector{Complex128}(length(basis(rho)))),
                tmp2=Ket(basis(rho), Vector{Complex128}(length(basis(rho)))))
    coherentstate(basis(rho), alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return a/pi
end

function qfunc(rho::Operator, X::Vector{Float64}, Y::Vector{Float64})
    b = basis(rho)
    Nx = length(X)
    Ny = length(Y)
    tmp1 = Ket(b, Vector{Complex128}(length(b)))
    tmp2 = Ket(b, Vector{Complex128}(length(b)))
    result = Matrix{Complex128}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        result[i, j] = qfunc(rho, complex(X[i], Y[j]), tmp1, tmp2)
    end
    return result
end

function qfunc(psi::Ket, alpha::Complex128)
    a = conj(alpha)
    N = length(psi.basis)
    s = psi.data[N]/sqrt(N-1)
    @inbounds for i=1:N-2
        s = (psi.data[N-i] + s*a)/sqrt(N-i-1)
    end
    s = psi.data[1] + s*a
    return abs2(s)*exp(-abs2(alpha))/pi
end

function _qfunc_ket(x::Vector{Complex128}, a::Complex128)
    s = x[1]
    @inbounds for i=2:length(x)
        s = x[i] + s*a
    end
    abs2(s)*exp(-abs2(a))/pi
end

function qfunc(psi::Ket, X::Vector{Float64}, Y::Vector{Float64})
    Nx = length(X)
    Ny = length(Y)
    N = length(psi.basis)
    n = 1.
    x = Vector{Complex128}(N)
    x[N] = psi.data[1]
    for i in 1:N-1
        x[N-i] = psi.data[i+1]/n
        n *= sqrt(i+1)
    end
    result = Matrix{Float64}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        a = complex(X[i], -Y[j])
        result[i, j] = _qfunc_ket(x, a)
    end
    return result
end

end # module
