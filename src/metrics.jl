module metrics

using ..operators, ..operators_dense

export tracedistance, tracedistance_general, entropy_vn

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


"""
Von Neumann entropy of density matrix.

The VN entropy of a density operator is defined as

.. math:

    S(\\rho) = -Tr(\\rho \\log(\\rho)) = -\\sum_n \\lambda_n\\log(\\lambda_n)

where :math:`\\lambda_n` are the eigenvalues of the density matrix
:math:`\\rho`, :math:`log` is the natural logarithm and :math:`\\log(0)\\equiv 0`.
"""
entropy_vn(rho::DenseOperator) = sum([d == 0 ? 0 : -d*log(d) for d=eigvals(rho.data)])

end # module
