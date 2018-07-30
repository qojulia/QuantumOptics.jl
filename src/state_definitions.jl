module state_definitions

export randstate, randoperator, thermalstate, coherentthermalstate, phase_average, passive_state

using ..bases, ..states, ..operators, ..operators_dense, ..fock
using LinearAlgebra

"""
    randstate(basis)

Calculate a random normalized ket state.
"""
function randstate(b::Basis)
    psi = Ket(b, rand(ComplexF64, length(b)))
    normalize!(psi)
    psi
end

"""
    randoperator(b1[, b2])

Calculate a random unnormalized dense operator.
"""
randoperator(b1::Basis, b2::Basis) = DenseOperator(b1, b2, rand(ComplexF64, length(b1), length(b2)))
randoperator(b::Basis) = randoperator(b, b)

"""
    thermalstate(H,T)

Thermal state ``exp(-H/T)/Tr[exp(-H/T)]``.
"""
function thermalstate(H::Operator,T::Real)
    return normalize(exp(-full(H)/T))
end

"""
    coherentthermalstate(basis::FockBasis,H,T,alpha)

Coherent thermal state ``D(α)exp(-H/T)/Tr[exp(-H/T)]D^†(α)``.
"""
function coherentthermalstate(basis::FockBasis,H::Operator,T::Real,alpha::Number)
    return displace(basis,alpha)*thermalstate(H,T)*dagger(displace(basis,alpha))
end

"""
    phase_average(rho)

Returns the phase-average of ``ρ`` containing only the diagonal elements.
"""
function phase_average(rho::DenseOperator)
    return DenseOperator(basis(rho),diagm(diag(rho.data)))
end

"""
    passive_state(rho,IncreasingEigenenergies::Bool=true)

Passive state ``π`` of ``ρ``. IncreasingEigenenergies=true means that higher indices correspond to higher energies.
"""
function passive_state(rho::DenseOperator,IncreasingEigenenergies::Bool=true)
    return DenseOperator(basis(rho),diagm(sort(abs.(eigvals(rho.data)),rev=IncreasingEigenenergies)))
end

end #module
