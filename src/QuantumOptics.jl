__precompile__()

module QuantumOptics

export bases, Basis, GenericBasis, CompositeBasis,
        tensor, ⊗,
        states, StateVector, Bra, Ket, basis_bra, basis_ket,
                dagger, normalize, normalize!,
        operators, Operator, DenseOperator, projector,
                expect, identityoperator, dense_identityoperator,
                ptrace, embed,
        operators_lazy, LazyOperator, LazyTensor, LazySum, LazyProduct,
        operators_sparse, SparseOperator,
                sparse_identityoperator,
        super, DenseSuperOperator, SparseSuperOperator,
                spre, spost, liouvillian,
        fock, FockBasis, number, destroy, create,
                fockstate, coherentstate, qfunc,
        spin, SpinBasis, sigmax, sigmay, sigmaz, sigmap, sigmam, spinup, spindown,
        subspace, SubspaceBasis, projector,
        particle, PositionBasis, MomentumBasis, gaussianstate,
                positionoperator, momentumoperator, laplace_x, laplace_p,
        nlevel, NLevelBasis, transition, nlevelstate,
        nparticlebasis, BosonicNParticleBasis, FermionicNParticleBasis, nparticleoperator_1, nparticleoperator_2,
        metrics, tracedistance,
        spectralanalysis, operatorspectrum, operatorspectrum_hermitian,
                eigenstates, eigenstates_hermitian, groundstate,
        timeevolution_simple,
        timeevolution,
        cumulantexpansion,
        steadystate,
        correlations


include("bases.jl")
include("states.jl")
include("operators.jl")
include("sparsematrix.jl")
include("operators_sparse.jl")
include("operators_lazy.jl")
include("superoperators.jl")
include("spin.jl")
include("fock.jl")
include("subspace.jl")
include("particle.jl")
include("nlevel.jl")
include("nparticles.jl")
include("metrics.jl")
include("ode_dopri.jl")
include("timeevolution_simple.jl")
module timeevolution
    include("master.jl")
    include("schroedinger.jl")
    include("mcwf.jl")
    using .timeevolution_master
    using .timeevolution_schroedinger
    using .timeevolution_mcwf
end
include("cumulantexpansion.jl")
include("steadystate.jl")
include("correlations.jl")
include("spectralanalysis.jl")
include("io.jl")


using .bases
using .states
using .operators
using .operators_sparse
using .operators_lazy
using .superoperators
using .spin
using .fock
using .subspace
using .particle
using .nlevel
using .nparticles
using .metrics
using .spectralanalysis


end # module
