module spin

export SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown

import Base: ==

using ..bases, ..states, ..operators, ..operators_sparse


"""
    SpinBasis(n)

Basis for spin-n particles.

The basis can be created for arbitrary spinnumbers by using a rational number,
e.g. `SpinBasis(3//2)`. The Pauli operators are defined for all possible
spin numbers.
"""
mutable struct SpinBasis <: Basis
    shape::Vector{Int}
    spinnumber::Rational{Int}
    function SpinBasis(spinnumber::Rational{Int})
        n = numerator(spinnumber)
        d = denominator(spinnumber)
        @assert d==2 || d==1
        @assert n > 0
        N = numerator(spinnumber*2 + 1)
        new([N], spinnumber)
    end
end
SpinBasis(spinnumber::Int) = SpinBasis(convert(Rational{Int}, spinnumber))

==(b1::SpinBasis, b2::SpinBasis) = b1.spinnumber==b2.spinnumber

"""
    sigmax(b::SpinBasis)

Pauli ``σ_x`` operator for the given Spin basis.
"""
function sigmax(b::SpinBasis)
    N = length(b)
    diag = ComplexF64[complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:N-1]
    data = spdiagm(1 => diag, -1 => diag)
    SparseOperator(b, data)
end

"""
    sigmay(b::SpinBasis)

Pauli ``σ_y`` operator for the given Spin basis.
"""
function sigmay(b::SpinBasis)
    N = length(b)
    diag = ComplexF64[1im*complex(sqrt(real((b.spinnumber + 1)*2*a - a*(a+1)))) for a=1:N-1]
    data = spdiagm(-1 => diag, 1 => -diag)
    SparseOperator(b, data)
end

"""
    sigmaz(b::SpinBasis)

Pauli ``σ_z`` operator for the given Spin basis.
"""
function sigmaz(b::SpinBasis)
    N = length(b)
    diag = ComplexF64[complex(2*m) for m=b.spinnumber:-1:-b.spinnumber]
    data = spdiagm(0 => diag)
    SparseOperator(b, data)
end

"""
    sigmap(b::SpinBasis)

Raising operator ``σ_+`` for the given Spin basis.
"""
function sigmap(b::SpinBasis)
    N = length(b)
    S = (b.spinnumber + 1)*b.spinnumber
    diag = ComplexF64[complex(sqrt(float(S - m*(m+1)))) for m=b.spinnumber-1:-1:-b.spinnumber]
    data = spdiagm(1 => diag)
    SparseOperator(b, data)
end

"""
    sigmam(b::SpinBasis)

Lowering operator ``σ_-`` for the given Spin basis.
"""
function sigmam(b::SpinBasis)
    N = length(b)
    S = (b.spinnumber + 1)*b.spinnumber
    diag = [complex(sqrt(float(S - m*(m-1)))) for m=b.spinnumber:-1:-b.spinnumber+1]
    data = spdiagm(-1 => diag)
    SparseOperator(b, data)
end


"""
    spinup(b::SpinBasis)

Spin up state for the given Spin basis.
"""
spinup(b::SpinBasis) = basisstate(b, 1)

"""
    spindown(b::SpinBasis)

Spin down state for the given Spin basis.
"""
spindown(b::SpinBasis) = basisstate(b, b.shape[1])


end #module
