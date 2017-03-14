module random

using ..bases, ..states, ..operators_dense

export randstate, randoperator

"""
Calculate a random normalized ket state.
"""
function randstate(b::Basis)
    psi = Ket(b, rand(Complex128, length(b)))
    normalize!(psi)
    psi
end

"""
Calculate a random unnormalized dense operator.
"""
randoperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, rand(Complex128, length(b1), length(b2)))
randoperator(b::Basis) = randoperator(b, b)

end #module
