__precompile__()

module QuantumOptics

export bases, Basis, GenericBasis, CompositeBasis,
        tensor, ⊗, permutesystems,
        states, StateVector, Bra, Ket, basis_bra, basis_ket,
                dagger, normalize, normalize!,
        operators, Operator, expect, identityoperator, ptrace, embed,
        operators_dense, DenseOperator, projector,
        operators_sparse, SparseOperator, diagonaloperator,
        operators_lazy, lazy, LazyWrapper,
                LazyTensor, LazySum, LazyProduct,
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
        metrics, tracenorm, tracenorm_general, tracedistance, tracedistance_general,
                entropy_vn, fidelity,
        spectralanalysis, operatorspectrum, operatorspectrum_hermitian,
                eigenstates, eigenstates_hermitian, groundstate, simdiag,
        timeevolution_simple,
        timeevolution, diagonaljumps,
        cumulantexpansion,
        correlationexpansion,
        steadystate,
        correlations


include("sortedindices.jl")
include("bases.jl")
include("states.jl")
include("operators.jl")
include("operators_dense.jl")
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
diagonaljumps = timeevolution.timeevolution_mcwf.diagonaljumps
include("cumulantexpansion.jl")
include("correlationexpansion.jl")
include("steadystate.jl")
include("correlations.jl")
include("spectralanalysis.jl")

using .bases
using .states
using .operators
using .operators_dense
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
