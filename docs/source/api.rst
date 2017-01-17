.. _section-api:

API
===

.. _section-api-bases:

Bases
-----

.. jl:autoabstract:: bases.jl Basis

.. jl:autotype:: bases.jl GenericBasis

.. jl:autofunction:: bases.jl length

.. jl:autofunction:: bases.jl multiplicable


Composite bases
^^^^^^^^^^^^^^^

.. jl:autotype:: bases.jl CompositeBasis

.. jl:autofunction:: bases.jl tensor

.. jl:autofunction:: bases.jl ptrace


Subspace bases
^^^^^^^^^^^^^^

.. jl:autotype:: subspace.jl SubspaceBasis

.. jl:autofunction:: subspace.jl orthonormalize

.. jl:autofunction:: subspace.jl projector



.. _section-api-states:

States
------

.. jl:autoabstract:: states.jl StateVector

.. jl:autotype:: states.jl Bra

.. jl:autotype:: states.jl Ket

.. jl:autofunction:: states.jl tensor

.. jl:autofunction:: operators.jl tensor(::Ket, ::Bra)

.. jl:autofunction:: operators.jl ptrace(::Ket, )

.. jl:autofunction:: operators.jl ptrace(::Bra, )

.. jl:autofunction:: states.jl dagger

.. jl:autofunction:: states.jl norm

.. jl:autofunction:: states.jl normalize

.. jl:autofunction:: states.jl normalize!

.. jl:autofunction:: states.jl basis_bra

.. jl:autofunction:: states.jl basis_ket



.. _section-api-operators:

Operators
---------

.. jl:autoabstract:: operators.jl Operator

.. jl:autofunction:: operators.jl tensor(a::DenseOperator, b::DenseOperator)

.. jl:autofunction:: operators.jl tensor(ops...)

.. jl:autofunction:: operators.jl dagger

.. jl:autofunction:: operators.jl projector

.. jl:autofunction:: operators.jl norm

.. jl:autofunction:: operators.jl trace

.. jl:autofunction:: operators.jl normalize

.. jl:autofunction:: operators.jl normalize!

.. jl:autofunction:: operators_sparse.jl sparse_identityoperator

.. jl:autofunction:: operators.jl expect

.. jl:autofunction:: operators.jl embed

.. jl:autofunction:: operators.jl ptrace(::DenseOperator, indices)

.. jl:autofunction:: operators.jl ptrace(::DenseOperator, index)

.. jl:autofunction:: operators.jl gemv!

.. jl:autofunction:: operators.jl gemm!


.. _section-api-denseoperators:

DenseOperators
^^^^^^^^^^^^^^

.. jl:autotype:: operators.jl DenseOperator

.. jl:autofunction:: operators.jl DenseOperator

.. jl:autofunction:: operators.jl full

.. jl:autofunction:: operators.jl dense_identityoperator


.. _section-api-sparseoperators:

SparseOperators
^^^^^^^^^^^^^^^

.. jl:autotype:: operators_sparse.jl SparseOperator

.. jl:autofunction:: operators_sparse.jl SparseOperator

.. jl:autofunction:: operators_sparse.jl sparse


.. _section-api-lazyoperators:

LazyOperators
^^^^^^^^^^^^^

.. jl:autoabstract:: operators_lazy.jl LazyOperator

.. jl:autotype:: operators_lazy.jl LazyTensor

.. jl:autotype:: operators_lazy.jl LazySum

.. jl:autotype:: operators_lazy.jl LazyProduct



.. _section-api-superoperators:

Superoperators
--------------

.. jl:autoabstract:: superoperators.jl SuperOperator

.. jl:autotype:: superoperators.jl DenseSuperOperator

.. jl:autotype:: superoperators.jl SparseSuperOperator

.. jl:autofunction:: superoperators.jl spre

.. jl:autofunction:: superoperators.jl spost

.. jl:autofunction:: superoperators.jl liouvillian

.. jl:autofunction:: superoperators.jl expm



.. section-api-metrics:

Metrics
-------

.. jl:autofunction:: metrics.jl tracedistance

.. jl:autofunction:: metrics.jl tracedistance_general



Systems
-------


.. _section-api-fock:

Fock
^^^^

.. jl:autotype:: fock.jl FockBasis

.. jl:autofunction:: fock.jl FockBasis

.. jl:autofunction:: fock.jl number

.. jl:autofunction:: fock.jl destroy

.. jl:autofunction:: fock.jl create

.. jl:autofunction:: fock.jl fockstate

.. jl:autofunction:: fock.jl coherentstate

.. jl:autofunction:: fock.jl qfunc


.. _section-api-nlevel:

N-level
^^^^^^^

.. jl:autotype:: nlevel.jl NLevelBasis

.. jl:autofunction:: nlevel.jl transition

.. jl:autofunction:: nlevel.jl nlevelstate


.. _section-api-spin:

Spin
^^^^

.. jl:autotype:: spin.jl SpinBasis

.. jl:autofunction:: spin.jl sigmax

.. jl:autofunction:: spin.jl sigmay

.. jl:autofunction:: spin.jl sigmaz

.. jl:autofunction:: spin.jl sigmap

.. jl:autofunction:: spin.jl sigmam

.. jl:autofunction:: spin.jl spinup

.. jl:autofunction:: spin.jl spindown


.. _section-api-particle:

Particle
^^^^^^^^

.. jl:autotype:: particle.jl PositionBasis

.. jl:autotype:: particle.jl MomentumBasis

.. jl:autofunction:: particle.jl spacing

.. jl:autofunction:: particle.jl samplepoints

.. jl:autofunction:: particle.jl positionoperator

.. jl:autofunction:: particle.jl momentumoperator

.. jl:autofunction:: particle.jl laplace_x

.. jl:autofunction:: particle.jl laplace_p

.. jl:autofunction:: particle.jl gaussianstate

.. jl:autotype:: particle.jl FFTOperator

.. jl:autofunction:: particle.jl FFTOperator


.. _section-api-nparticles:

N-Particles
^^^^^^^^^^^

.. jl:autoabstract:: nparticles.jl NParticleBasis

.. jl:autotype:: nparticles.jl BosonicNParticleBasis

.. jl:autotype:: nparticles.jl FermionicNParticleBasis

.. jl:autofunction:: nparticles.jl nparticleoperator_1

.. jl:autofunction:: nparticles.jl nparticleoperator_2



.. _section-api-timeevolution:

Time-evolution
--------------

.. _section-api-schroedinger:


Schroedinger
^^^^^^^^^^^^

.. jl:autofunction:: schroedinger.jl schroedinger


.. _section-api-master:

Master
^^^^^^

.. jl:autofunction:: master.jl master

.. jl:autofunction:: master.jl master_h

.. jl:autofunction:: master.jl master_nh


.. _section-api-mcwf:

Monte Carlo wave function
^^^^^^^^^^^^^^^^^^^^^^^^^

.. jl:autofunction:: mcwf.jl mcwf



.. _section-api-spectralanalysis:

Spectral analysis
-----------------

.. jl:autofunction:: spectralanalysis.jl operatorspectrum_hermitian

.. jl:autofunction:: spectralanalysis.jl operatorspectrum

.. jl:autofunction:: spectralanalysis.jl eigenstates_hermitian

.. jl:autofunction:: spectralanalysis.jl eigenstates

.. jl:autofunction:: spectralanalysis.jl groundstate


.. _section-api-steadystate:

Steady-states
-------------

.. jl:autofunction:: steadystate.jl master

.. jl:autofunction:: steadystate.jl eigenvector


.. _section-api-correlations:

Correlations
------------

.. jl:autofunction:: correlations.jl correlation

.. jl:autofunction:: correlations.jl correlationspectrum
