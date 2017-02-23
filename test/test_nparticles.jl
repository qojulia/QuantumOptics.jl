using Base.Test
using QuantumOptics

@testset "nparticles" begin

particlenumber = 3
modenumber = 3
b_spin = SpinBasis(1//2)

m = complex(randn(modenumber, modenumber))
b = BosonicNParticleBasis(particlenumber, modenumber)
op = DenseOperator(GenericBasis([modenumber]), m)
op2 = SparseOperator(GenericBasis([modenumber]), sparse(m))
op_ = nparticleoperator_1(b, op)
op2_ = nparticleoperator_1(b, op2)
op_ = nparticleoperator_1(b, op)
op2_ = nparticleoperator_1(b, op2)

@test tracedistance(op_, full(op2_)) < 1e-12


m = complex(randn(modenumber^2, modenumber^2))
b = BosonicNParticleBasis(particlenumber, modenumber)
op = DenseOperator(GenericBasis([modenumber^2]), m)
op2 = SparseOperator(GenericBasis([modenumber^2]), sparse(m))
op_ = nparticleoperator_2(b, op)
op2_ = nparticleoperator_2(b, op2)
op_ = nparticleoperator_2(b, op)
op2_ = nparticleoperator_2(b, op2)

@test tracedistance(op_, full(op2_)) < 1e-12

end # testset
