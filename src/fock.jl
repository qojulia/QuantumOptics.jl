module fock

import Base.==

using ..bases, ..states, ..operators, ..operators_sparse

export FockBasis, number, destroy, create, fockstate, coherentstate, qfunc


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
    basis_ket(b, n+1-b.Nmin)
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
function coherentstate(b::FockBasis, alpha::Number, result=Ket(b, Vector{Complex128}(b.shape[1])))
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
                tmp1=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1])),
                tmp2=Ket(rho.basis_l, Vector{Complex128}(rho.basis_l.shape[1])))
    b = rho.basis_l
    coherentstate(b, alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return real(a)/pi
end

function qfunc(rho::Operator, X::Vector{Float64}, Y::Vector{Float64})
    @assert rho.basis_l == rho.basis_r
    b = rho.basis_l
    Nx = length(X)
    Ny = length(Y)
    tmp1 = Ket(b, Vector{Complex128}(b.shape[1]))
    tmp2 = Ket(b, Vector{Complex128}(b.shape[1]))
    result = Matrix{Float64}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        result[i, j] = qfunc(rho, complex(X[i], Y[j]), tmp1, tmp2)
    end
    return result
end

end # module
