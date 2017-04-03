.. _section-schroedinger:

Schroedinger time evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Schroedinger equation as one of the basic postulates of quantum mechanics describes the dynamics of a quantum state in a closed quantum system. In Dirac notation the Schroedinger equation and its adjoint equation read

.. math::

    i\hbar\frac{\mathrm{d}}{\mathrm{d} t} |\Psi(t)\rangle = H |\Psi(t)\rangle

    - i\hbar\frac{\mathrm{d}}{\mathrm{d} t} \langle \Psi(t)| = \langle\Psi(t)| H

Both versions are implemented and are chosen automatically depending on the type of the provided initial state (Bra or Ket)::

  schroedinger(tspan::Vector{Float64}, psi0::{Ket,Bra}, H::Operator)


The Schrödinger equation solver requires the arguments ``tspan``, which is a vector containing the times, the initial state ``psi0`` as ``Ket`` or ``Bra`` and the Hamiltonian ``H``.

Additionally, one can pass an output function ``fout`` as keyword argument. This can be convenient if one directly wants to compute a value that depends on the states, e.g. an expectation value, instead of the states themselves. Consider, for example, a time evolution according to a Schrödinger equation where for all times we want to compute the expectation value of the operator ``A``. We can do this by::

    tout, psi_t = timeevolution.schroedinger(T, psi0, H)
    exp_val = expect(A, psi_t)

or equivalently::

    tout = Float64[]
    exp_val = Complex128[]
    function exp(t, psi)
      push!(tout, t)
      push!(exp_val, expect(A, psi)
    end
    timeevolution.schroedinger(T, psi0, H; fout=exp)

Although the method using ``fout`` might seem more complicated, it can be very useful for large systems to save memory since instead of all the states we only store one complex number per time step. Note, that ``fout`` must always be defined with the arguments ``(t, psi)``. If ``fout`` is given, all variables are assigned within ``fout`` and the call to :jl:func:`schroedinger` returns ``nothing``.

We can also calculate the time evolution for a Hamiltonian that is time-dependent. In that case, we need to use the function :jl:func:`schroedinger_dynamic(tspan, psi0, f::Function)`. As you can see, this function requires the same arguments as :jl:func:`schroedinger`, but a function ``f`` instead of a Hamiltonian. As a brief example, consider a spin-1/2 particle that is coherently driven by a laser that has an amplitude that varies in time. We can implement this with::

  basis = SpinBasis(1//2)
  ψ₀ = spindown(basis)
  function pump(t, psi)
    return sin(t)*(sigmap(basis) + sigmam(basis))
  end
  tspan = [0:0.1:10;]
  tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, pump)


Examples
--------

* :ref:`example-jaynes-cummings`
* :ref:`example-particle-in-harmonic-trap`
* :ref:`example-particle-into-barrier`
* :ref:`example-two-qubit-entanglement`
* :ref:`example-nparticles-in-double-well`
