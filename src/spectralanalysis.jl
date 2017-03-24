module spectralanalysis

import Base.eig, Base.eigs, Base.eigvals, Base.eigvals!
using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export simdiag


"""
Diagonalize an operator.

Arguments
---------

A
    Sparse or dense operator.

Keyword arguments
-----------------
See http://docs.julialang.org/en/stable/stdlib/linalg/

args (optional)
  Aditional arguments of Julia's eig function.

kwargs (optional)
  Additional kwargs of Julia's eigs function used for sparse operators.

Returns
-------

D
  Vector of eigenvalues sorted from smalles (abs) to largest.
V
  Vector of Kets in the basis of A sorted by D.
"""
function eig(A::DenseOperator, args...)
  check_samebases(A)
  b = A.basis_l
  if ishermitian(A)
    D, V = eig(Hermitian(A.data), args...)
    states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
  else
    D, V = eig(A.data, args...)
    states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
    perm = sortperm(D, by=abs)
    permute!(D, perm)
    permute!(states, perm)
  end
  return D, states
end

function eigs(A::SparseOperator, args...; kwargs...)
  check_samebases(A)
  b = A.basis_l
  if ishermitian(A)
    D, V = eigs(Hermitian(A.data), args...; kwargs...)
  else
    D, V = eigs(A.data, args...; kwargs...)
    states = [Ket(A.basis_l, V[:, k]) for k=1:length(D)]
    perm = sortperm(D, by=abs)
    permute!(D, perm)
    permute!(states, perm)
  end
  return D, states
end

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
See http://docs.julialang.org/en/stable/stdlib/linalg/

args (optional)
  Aditional arguments of Julia's eigvals function.

Returns
-------
Vector containing eigenvalues sorted from smallest to largest absolute value.
"""
eigvals(A::DenseOperator, args...) = ishermitian(A) ? eigvals(Hermitian(A.data), args...) : sort(eigvals(A.data), by=abs)
eigvals!(A::DenseOperator, args...) = ishermitian(A) ? eigvals!(Hermitian(A.data), args...) : sort(eigvals!(A.data), by=abs)

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

Returns
-------

evals_sorted
  Vector containing all vectors of the eigenvalues sorted by the eigenvalues
  of the first operator.
v
  Common eigenvectors.
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
  return evals_sorted, v[:, index]
end

end # module
