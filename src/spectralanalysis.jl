module spectralanalysis

import Base.eig, Base.eigs, Base.eigvals, Base.eigvals!
using ..states, ..operators, ..operators_dense, ..operators_sparse

export simdiag


"""
Diagonalize an operator.

Arguments
---------

A
    Sparse or dense operator.

Keyword arguments
-----------------

args (optional)
  Aditional arguments of Julia's eig function. Have to be given in the following order.

  irange (requires Hermitian)
    UnitRange that specifies a range of indices for which the eigenvalues and eigenvectors
    of a Hermitian operator are calculated.

  vl (requires Hermitian)
    Lower boundary for calculation of eigenvalues.

  vu (requires Hermitian)
    Upper boundary for calculation of eigenvalues. Only eigenvalues :math:`\\lambda`
    (and their corresponding eigenvectors) for which :math:`vl \\leq \\lambda \\leq vu`
    are computed.

  permute
    Boolean (default is :func:`true`), that if true allows permutation of the matrix so
    it becomes closer to a upper diagonal matrix

  scale
    Boolean (default is :func:`true`), that if true scales the matrix by its diagonal
    so they are closer in norm.


"""
eig(A::DenseOperator, args...) = ishermitian(A) ? eig(Hermitian(A.data), args...) : eig(A.data, args...)
eigs(A::SparseOperator, args...) = ishermitian(A) ? eigs(Hermitian(A.data), args...) : eigs(A.data, args...)

arithmetic_unary_error = operators.arithmetic_unary_error
eig(A::Operator, args...) = arithmetic_unary_error(eig, A)
eigs(A::Operator, args...) = arithmetic_unary_error(eig, A)


"""
Compute eigenvalues of an operator.

The in-place calculation can be used with :func:`eigenvalues!`.

Arguments
---------

A
    Sparse or dense operator.

Keyword arguments
-----------------

args (optional)
  Aditional arguments of Julia's eig function. Have to be given in the following order.

  irange (requires Hermitian)
    UnitRange that specifies a range of indices for which the eigenvalues and eigenvectors
    of a Hermitian operator are calculated.

  vl (requires Hermitian)
    Lower boundary for calculation of eigenvalues.

  vu (requires Hermitian)
    Upper boundary for calculation of eigenvalues. Only eigenvalues :math:`\\lambda`
    (and their corresponding eigenvectors) for which :math:`vl \\leq \\lambda \\leq vu`
    are computed.

"""
eigvals(A::DenseOperator, args...) = ishermitian(A) ? eigvals(Hermitian(A.data), args...) : eigvals(A.data)
eigvals(A::SparseOperator, args...) = eigvals(full(A), args...)
eigvals!(A::DenseOperator, args...) = ishermitian(A) ? eigvals!(Hermitian(A.data), args...) : eigvals!(A.data)
eigvals!(A::SparseOperator, args...) = eigvals!(full(A))

eigvals(A::Operator, args...) = arithmetic_unary_error(eigvals, A)
eigvals!(A::Operator, args...) = arithmetic_unary_error(eigvals!, A)


"""
Simultaneously diagonalize commuting Hermitian operators.

This is done by diagonalizing the sum of the operators.
The eigenvalues are computed by :math:`a = \\langle \\psi |A|\\psi\\rangle` and
it is checked whether the eigenvectors fulfill the equation
:math:`A|\\psi\\rangle = a|\\psi\\rangle`.

Arguments
---------

Ops
  Vector of operators (sparse or dense).

Keyword arguments
-----------------

atol (optional)
  kwarg of Base.isapprox specifying the tolerance of the approximate check
  Default is 1e-14.

rtol (optional)
  kwarg of Base.isapprox specifying the tolerance of the approximate check
  Default is 1e-14.
"""
function simdiag{T <: DenseOperator}(Ops::Vector{T}; atol::Real=1e-14, rtol::Real=1e-14)

  # Check input
  for A=Ops
    if !ishermitian(A)
      error("Non-hermitian operator given!")
    end
  end

  d, v = eig(sum(Ops).data)

  evals = [Vector{Complex128}(length(d)) for i=1:length(Ops)]
  for i=1:length(Ops), j=1:length(d)
    vec = Ops[i].data*v[:, j]
    evals[i][j] = (v[:, j]'*vec)[1]
    if !isapprox(vec, evals[i][j]*v[:, j]; atol=atol, rtol=rtol)
      error("Simultaneous diagonalization failed!")
    end
  end

  index = sortperm(real(evals[1][:]))
  evals_sorted = [real(evals[i][index]) for i=1:length(Ops)]
  evals_sorted, v[:, index]
end

end # module
