module fock

import Base.==

using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export FockBasis, number, destroy, create, fockstate, coherentstate, qfunc, displace


"""
Basis for a Fock space.

Arguments
---------

Nmin
    Minimal particle number included in this basis. By default it is zero, i.e.
    starting with the vacuum state.
Nmax
    Maximal particle number included in this basis.
"""
type FockBasis <: Basis
    shape::Vector{Int}
    Nmin::Int
    Nmax::Int
    function FockBasis(Nmin::Int, Nmax::Int)
        if Nmin < 0 || Nmax <= Nmin
            throw(DimensionMismatch())
        end
        new([Nmax-Nmin+1], Nmin, Nmax)
    end
end

FockBasis(Nmax::Int) = FockBasis(0, Nmax)


==(b1::FockBasis, b2::FockBasis) = b1.Nmin==b2.Nmin && b1.Nmax==b2.Nmax

"""
Number operator for the given Fock space.

Arguments
---------

b
    FockBasis of this operator.
"""
function number(b::FockBasis)
    N = b.Nmax-b.Nmin+1
    diag = Complex128[complex(x) for x=b.Nmin:b.Nmax]
    data = spdiagm(diag, 0, N, N)
    SparseOperator(b, data)
end

"""
Annihilation operator for the given Fock space.

Arguments
---------

b
    FockBasis of this operator.
"""
function destroy(b::FockBasis)
    N = b.Nmax-b.Nmin+1
    diag = Complex128[complex(sqrt(x)) for x=b.Nmin+1:b.Nmax]
    data = spdiagm(diag, 1, N, N)
    SparseOperator(b, data)
end

"""
Creation operator for the given Fock space.

Arguments
---------

b
    FockBasis of this operator.
"""
function create(b::FockBasis)
    N = b.Nmax-b.Nmin+1
    diag = Complex128[complex(sqrt(x)) for x=b.Nmin+1:b.Nmax]
    data = spdiagm(diag, -1, N, N)
    SparseOperator(b, data)
end

"""
Fock state for the given particle number.

Arguments
---------

b
    FockBasis for this state.
n
    Quantum number of the state.
"""
function fockstate(b::FockBasis, n::Int)
    @assert b.Nmin <= n <= b.Nmax
    basisstate(b, n+1-b.Nmin)
end

"""
Coherent state :math:`| \\alpha \\rangle` for the given Fock basis.

Arguments
---------

b
    FockBasis for this state.
alpha
    Eigenvalue of annihilation operator.
"""
function coherentstate(b::FockBasis, alpha::Number, result=Ket(b, Vector{Complex128}(length(b))))
    alpha = complex(alpha)
    data = result.data
    if b.Nmin == 0
        data[1] = exp(-abs2(alpha)/2)
    else
        data[1] = exp(-abs2(alpha)/2)*alpha^b.Nmin/sqrt(factorial(b.Nmin))
    end
    @inbounds for n=1:(b.Nmax-b.Nmin)
        data[n+1] = data[n]*alpha/sqrt(b.Nmin + n)
    end
    return result
end

"""
Husimi Q representation :math:`\\frac{1}{\\pi} \\langle \\alpha | \\rho | \\alpha \\rangle`.
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

"""
Construct displacement operator.
"""
displace(b::FockBasis, α::Complex128) = expm(full(α*create(b) - conj(α)*destroy(b)))

end # module
