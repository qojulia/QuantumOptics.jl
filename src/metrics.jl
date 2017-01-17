module metrics

using ..operators, ..operators_sparse

export tracedistance, tracedistance_general

"""
Trace distance between two density operators.

It uses the identity

.. math:

    T(\\rho, \\sigma) = \\frac{1}{2} \\sum_i |\\lambda_i|

where :math:`\\lambda_i` are the eigenvalues of the matrix
:math:`\\rho - \\sigma`. This works only if :math:`rho` and :math:`sigma`
are density operators. For trace distances between general operators use
:jl:func:`tracedistance_general`.
"""
function tracedistance(rho::DenseOperator, sigma::DenseOperator)
    delta = (rho - sigma)
    @assert length(delta.basis_l) == length(delta.basis_r)
    data = delta.data
    for i=1:size(data,1)
        data[i,i] = real(data[i,i])
    end
    s = eigvals(Hermitian(data))
    return 0.5*sum(abs(s))
end


"""
Trace distance between two operators.

.. math:

    T(\\rho, \\sigma) = \\frac{1}{2}
            Tr\\{\\ \\sqrt{(\\rho-\\sigma)^\\dagger(\\rho-\\sigma)}\\}
"""
function tracedistance_general(rho::DenseOperator, sigma::DenseOperator)
    delta = (rho - sigma)
    return 0.5*trace(sqrtm((dagger(delta)*delta).data))
end

function tracedistance{T<:Operator}(rho::T, sigma::T)
    throw(ArgumentError("tracedistance not implemented for $(T). Use dense operators instead."))
end

end # module
