.. _tutorial:

Tutorial
========

In this tutorial we will step through the common tasks necessary to simulate a quantum system. At each step links to documentation and examples will be provided that explain the topic at hand in more detail.

In order to use the **QuantumOptics.jl** library it has to be loaded into the current workspace.

.. code-block:: julia

    using QuantumOptics


Bases
-----

The first step is always to define a suitable basis for the quantum system under consideration. For example a Fock basis can be used to describe a field mode. It takes an integer as argument which specifies the photon number cutoff.

.. code-block:: julia

    b = FockBasis(20)

Many common quantum systems are already implemented:

* :ref:`section-spin`
* :ref:`section-fock`
* :ref:`section-particle`
* :ref:`section-nlevel`
* :ref:`section-manybody`


States and Operators
--------------------

Most of the typical operators and states in these bases are defined.

.. code-block:: julia

    a = destroy(b)
    at = create(b)

    alpha = 1.5
    psi = coherentstate(b, alpha)

These objects can be combined in all the expected ways.

.. code-block:: julia

    dpsi = -1im*(a + at)*psi

The :jl:func:`dagger()` function can be used to transform a ket state into a bra state which for example can be used to create a density operator

.. code-block:: julia

    rho = psi ⊗ dagger(psi)

or to calculate expectation values

.. code-block:: julia

    >>> println(dagger(psi)*a*psi)
    1.4999999999991955 + 0.0im

Alternatively, the :jl:func:`expect()` function can be used which is faster and also directly extends to the case where the state is not a ket but a density operator.

.. code-block:: julia

    >>> println(expect(a, psi))
    1.4999999999991955 + 0.0im


Composite systems
-----------------

Most interesting quantum systems consist of several different parts, for example a spin coupled to a cavity mode.

.. code-block:: julia

    ω_atom = 2
    ω_field = 1

    # 2 level atom described as spin
    b_spin = SpinBasis(1//2)
    sp = sigmap(b_spin)
    sm = sigmam(b_spin)

    H_atom = ω_atom*sp*sm

    # Use a Fock basis with a maximum of 20 photons to model a cavity mode
    b_fock = FockBasis(20)
    a = destroy(b_fock)
    at = create(b_fock)
    n = number(b_fock)

    H_field = ω_field*n

Combining operators from those two systems can be done with the :jl:func:`tensor()` function or with the equivalent :math:`\otimes` operator.

.. code-block:: julia

    Ω = 1
    H_int = Ω*(a ⊗ sp + at ⊗ sm)

To extend the single system Hamiltonians :math:`H_{atom}` and :math:`H_{spin}` to the composite system Hilbert space, one possibility is to combine them with identity operators from the opposite sub-system.

.. code-block:: julia

    I_field = identityoperator(b_fock)
    I_atom = identityoperator(b_spin)

    H_atom_ = I_field ⊗ H_atom
    H_field_ = I_atom ⊗ H_field

However, especially for larger systems this can become tedious and it's more convenient to use the :jl:func:`embed()` function.

.. code-block:: julia

    b = b_fock ⊗ b_spin # Basis of composite system

    H = embed(b, 1, H_field) + embed(b, 2, H_atom) + H_int

Creating composite states works equivalently.

.. code-block:: julia

    ψ0 = fockstate(b_fock, 1) ⊗ spindown(b_spin)

Time evolution
--------------

Several different types of :ref:`time evolution <section-timeevolution>` are implemented in **QuantumOptics.jl**:

* :ref:`section-schroedinger`
* :ref:`section-master`
* :ref:`section-mcwf`

All of them share a very similar interface so that changing from one to another is mostly done by exchanging the names:

.. code-block:: julia

    schroedinger(T, psi0, H)
    master(T, psi0/rho0, H, J)
    mcwf(T, psi0, H, J)

Let's now simulate the dynamics of the system according to the Schrödinger equation.

.. code-block:: julia

    T = [0:0.05:5;]
    tout, ψt = timeevolution.schroedinger(T, ψ0, H)

Adding photon loss to the cavity by introducing a jump operator :math:`a` means that the system is an open quantum system and its time evolution is then described by a master equation.

.. code-block:: julia

    κ = 1.
    J = [embed(b, 1, a)]
    tout, ρt = timeevolution.master(T, ψ0, H, J; Gamma=[κ])

Alternatively, one can use the MCWF method to analyze the time evolution of the system. Physically, it can be interpreted as an experimental setup where every photon leaving the cavity is meassured by a photon counter, thereby projecting the system onto the state :math:`| \psi\rangle \rightarrow a |\psi\rangle`. This leads to a coherent time evolution according to a Schrödinger equation interrupted by jumps at certain random points in time.

.. code-block:: julia

    tout, ψt_mcwf = timeevolution.mcwf(T, ψ0, H, J; seed=UInt(0),
                                       display_beforeevent=true,
                                       display_afterevent=true)


The results can be visualized using for example `Matplotlib <matplotlib.org>`_ via `PyPlot.jl <https://github.com/JuliaPy/PyPlot.jl>`_.

.. code-block:: julia

    using PyPlot

    figure(figsize=[10, 3])

    subplot(1, 2, 1)
    xlabel("Time")
    ylabel(L"$\langle |e\rangle \langle e| \rangle$")
    xlim(0, 5)
    ylim(0, 1)
    plot(T, expect(2, sp*sm, ψt), label="Schrödinger")
    plot(T, expect(2, sp*sm, ρt), label="Master")
    plot(tout, expect(2, sp*sm, ψt_mcwf), label="MCWF")
    legend()

    subplot(1, 2, 2)
    xlabel("Time")
    ylabel(L"$\langle n \rangle$")
    xlim(0, 5)
    ylim(0, 1)
    plot(T, expect(1, n, ψt), label="Schrödinger")
    plot(T, expect(1, n, ρt), label="Master")
    plot(tout, expect(1, n, ψt_mcwf), label="MCWF")
    legend();

.. image:: _tutorial_files/_tutorial_34_0.png


In the statistical average the MCWF time evolution is equivalent to the
time evolution according to the master equation.

.. code-block:: julia

    Ntrajectories = 10

    exp_n = zeros(Float64, length(T))
    exp_e = zeros(Float64, length(T))

    function fout(t, psi)
        i = findfirst(T, t)
        N = norm(psi)
        exp_e[i] += real(expect(2, sp*sm, normalize(psi)))
        exp_n[i] += real(expect(1, n, normalize(psi)))
    end

    srand(0)
    for i=1:Ntrajectories
        timeevolution.mcwf(T, ψ0, H, J; fout=fout)
    end

    figure(figsize=[10, 3])

    subplot(1, 2, 1)
    xlabel("Time")
    ylabel(L"$\langle |e\rangle \langle e| \rangle$")
    plot(T, expect(2, sp*sm, ρt), label="Master")
    plot(T, exp_e/Ntrajectories, label=L"$\langle \mathrm{MCWF} \rangle$")
    legend()

    subplot(1, 2, 2)
    xlabel("Time")
    ylabel(L"$\langle n \rangle$")
    plot(T, expect(1, n, ρt), label="Master")
    plot(T, exp_n/Ntrajectories, label=L"$\langle \mathrm{MCWF} \rangle$")
    legend();

.. image:: _tutorial_files/_tutorial_36_0.png
