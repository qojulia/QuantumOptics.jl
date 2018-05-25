module state_definitions

export randstate, randoperator, thermalstate

using ..bases, ..states, ..operators, ..operators_dense


"""
    randstate(basis)

Calculate a random normalized ket state.
"""
function randstate(b::Basis)
    psi = Ket(b, rand(Complex128, length(b)))
    normalize!(psi)
    psi
end

"""
    randoperator(b1[, b2])

Calculate a random unnormalized dense operator.
"""
randoperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, rand(Complex128, length(b1), length(b2)))
randoperator(b::Basis) = randoperator(b, b)

"""
    thermalstate(H,T)

Thermal state ``exp(-H/T)/Tr[exp(-H/T)]``.
"""
function thermalstate(H::Operator,T::Real)
    return normalize(expm(-full(H)/T))
end

end #module
