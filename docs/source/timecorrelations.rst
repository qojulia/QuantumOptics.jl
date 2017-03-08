.. _section-correlationfunctions:

Two-time correlation functions
==============================

.. jl:autofunction:: correlations.jl correlation

As a brief example, say we want to calculate the two-time correlation function of a cavity field, i.e. :math:`g(t) = \langle a(t) a^\dagger(0)`.
We can do this with::

  basis = FockBasis(10)
  a = destroy(basis)
  κ = 1.0
  η = 0.3
  H = η*(a + dagger(a))
  J = [sqrt(2κ)*a]

  tspan = [0:0.1:10;]
  ρ0 = fockstate(basis, 0) ⊗ dagger(fockstate(basis, 0))
  g = correlations.correlation(tspan, ρ0, H, J, dagger(a), a)

If we omit the list of times :func:`tspan`, the function automatically calculates the correlation until steady-state is reached::

  t_s, g_s = correlations.correlation(ρ0, H, J, dagger(a), a)
