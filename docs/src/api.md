# API


## [Quantum-objects](@id API: Quantum objects)

### [Types](@id API: Quantum objects types)

* General basis types. Specialized bases can be found in the section [API: Quantum-systems](@ref).

```@docs
Basis
```

```@docs
GenericBasis
```

```@docs
CompositeBasis
```

* States

```@docs
StateVector
```

```@docs
Bra
```

```@docs
Ket
```

```@docs
LazyKet
```

* General purpose operators. A few more specialized operators are implemented in [API: Quantum-systems](@ref).

```@docs
AbstractOperator
```

```@docs
DataOperator
```

```@docs
Operator
```

```@docs
DenseOperator
```

```@docs
SparseOperator
```

```@docs
LazyKet


```@docs
LazyTensor
```

```@docs
LazySum
```

```@docs
LazyProduct
```

* Time-dependent operators. See also [Time-dependent operators](@ref).

```@docs
AbstractTimeDependentOperator
```

```@docs
TimeDependentSum
```

* Super operators:

```@docs
SuperOperator
```

```@docs
DenseSuperOperator
```

```@docs
SparseSuperOperator
```


### [Functions](@id API: Quantum objects functions)

* Functions to generate general states, operators and super-operators

```@docs
basisstate
```

```@docs
sparsebasisstate
```

```@docs
identityoperator
```

```@docs
diagonaloperator
```

```@docs
randoperator
```

```@docs
randunitary_haar
```

```@docs
spre
```

```@docs
spost
```

```@docs
sprepost
```

```@docs
liouvillian
```

* As far as it makes sense the same functions are implemented for bases, states, operators and superoperators.

```@docs
QuantumOpticsBase.samebases
```

```@docs
QuantumOpticsBase.check_samebases
```

```@docs
@samebases
```

```@docs
QuantumOpticsBase.multiplicable
```

```@docs
QuantumOpticsBase.check_multiplicable
```

```@docs
QuantumOpticsBase.basis
```

```@docs
dagger
```

```@docs
tensor
```

```@docs
projector(a::Ket, b::Bra)
projector(a::Ket)
projector(a::Bra)
```

```@docs
dm
```

```@docs
norm(x::StateVector)
```

```@docs
tr
```

```@docs
ptrace
```

```@docs
normalize(x::StateVector)
normalize(op::AbstractOperator)
```

```@docs
normalize!(x::StateVector)
normalize!(op::AbstractOperator)
```

```@docs
expect
```

```@docs
variance
```

```@docs
embed
```

```@docs
permutesystems
```

```@docs
exp(op::AbstractOperator)
```

```@docs
QuantumOpticsBase.mul!
```

* Conversion of operators

```@docs
dense
```

```@docs
sparse(::AbstractOperator)
```

* Time-dependent operators. See also [Time-dependent operators](@ref).

```@docs
current_time
```

```@docs
static_operator
```

```@docs
set_time!
```

```@docs
time_shift
```

```@docs
time_stretch
```

```@docs
time_restrict
```

```@docs
static_operator
```

### [Exceptions](@id API: Quantum objects exceptions)

```@docs
QuantumOpticsBase.IncompatibleBases
```


## [Quantum systems](@id API: Quantum systems)

### [Fock](@id API: Fock)

```@docs
FockBasis
```

```@docs
number(::Type{T}, ::FockBasis) where T
```

```@docs
destroy(::Type{C}, ::FockBasis) where C
```

```@docs
create(::Type{C}, ::FockBasis) where C
```

```@docs
displace
```

```@docs
displace_analytical
```

```@docs
displace_analytical!
```

```@docs
fockstate
```

```@docs
coherentstate
```

```@docs
coherentstate!
```

```@docs
squeeze
```

### [Phase space](@id API: Phase space)

```@docs
qfunc
```

```@docs
wigner
```

```@docs
coherentspinstate
```

```@docs
qfuncsu2
```

```@docs
wignersu2
```

```@docs
ylm
```

### [Charge](@id API: Charge)

```@docs
ChargeBasis
```

```@docs
ShiftedChargeBasis
```

```@docs
chargestate
```

```@docs
chargeop
```

```@docs
expiφ
```

```@docs
cosφ
```

```@docs
sinφ
```

### [N-level](@id API: N-level)

```@docs
NLevelBasis
```

```@docs
transition(::Type{T}, ::NLevelBasis, ::Integer, ::Integer) where T
```

```@docs
nlevelstate
```

```@docs
paulix
```

```@docs
pauliy
```

```@docs
pauliz
```

### [Spin](@id API: Spin)

```@docs
SpinBasis
```

```@docs
sigmax
```

```@docs
sigmay
```

```@docs
sigmaz
```

```@docs
sigmap
```

```@docs
sigmam
```

```@docs
spinup
```

```@docs
spindown
```

### [Particle](@id API: Particle)

```@docs
PositionBasis
```

```@docs
MomentumBasis
```

```@docs
spacing
```

```@docs
samplepoints
```

```@docs
position(::Type{T}, b::PositionBasis) where T
position(::Type{T}, b::MomentumBasis) where T
```

```@docs
momentum(::Type{T}, b::PositionBasis) where T
momentum(::Type{T}, b::MomentumBasis) where T
```

```@docs
potentialoperator
```

```@docs
gaussianstate
```

```@docs
QuantumOpticsBase.FFTOperator
```

```@docs
QuantumOpticsBase.FFTOperators
```

```@docs
QuantumOpticsBase.FFTKets
```

```@docs
transform
```

### [Charge](@id API: Charge)

```@docs
ChargeBasis
```

```@docs
ShiftedChargeBasis
```

```@docs
chargestate
```

```@docs
chargeop
```

```@docs
expiφ
```

```@docs
cosφ
```

```@docs
sinφ
```

### [Subspace bases](@id API: Subspace bases)

```@docs
SubspaceBasis
```

```@docs
QuantumOpticsBase.orthonormalize
```

```@docs
projector(::Type{T}, b1::SubspaceBasis, b2::SubspaceBasis) where T
```

```@docs
sparseprojector
```

### [Many-body](@id API: Many-body)

```@docs
ManyBodyBasis
```

```@docs
FermionBitstring
```

```@docs
fermionstates
```

```@docs
FermionBitstring
```

```@docs
bosonstates
```

```@docs
number(::Type{T}, ::ManyBodyBasis, index) where T
number(::Type{T}, ::ManyBodyBasis) where T
```

```@docs
destroy(::Type{T}, ::ManyBodyBasis, index) where T
```

```@docs
create(::Type{T}, ::ManyBodyBasis, index) where T
```

```@docs
transition(::Type{T}, ::ManyBodyBasis, i, j) where T
```

```@docs
manybodyoperator
```

```@docs
onebodyexpect
```

## [Direct sum](@id API: Direct-sum)

```@docs
SumBasis
```

```@docs
directsum
```

```@docs
LazyDirectSum
```

```@docs
getblock
```

```@docs
setblock!
```

## [Metrics](@id API: Metrics)

```@docs
tracenorm
```

```@docs
tracenorm_h
```

```@docs
tracenorm_nh
```

```@docs
tracedistance
```

```@docs
tracedistance_h
```

```@docs
tracedistance_nh
```

```@docs
entropy_vn
```

```@docs
entropy_renyi
```

```@docs
fidelity
```

```@docs
ptranspose
```

```@docs
PPT
```

```@docs
negativity
```

```@docs
logarithmic_negativity
```

```@docs
entanglement_entropy
```

```@docs
avg_gate_fidelity
```


## [Time-evolution](@id API: Time-evolution)

```@docs
timeevolution.recast!
```

### [Schroedinger](@id API: Schroedinger)

```@docs
timeevolution.schroedinger
```

```@docs
timeevolution.schroedinger_dynamic
```

```@docs
timeevolution.dschroedinger!
```

```@docs
timeevolution.dschroedinger_dynamic!
```

### [Master](@id API: Master)

```@docs
timeevolution.master
```

```@docs
timeevolution.master_h
```

```@docs
timeevolution.master_nh
```

```@docs
timeevolution.master_dynamic
```

```@docs
timeevolution.master_nh_dynamic
```

```@docs
timeevolution.dmaster_h!
```

```@docs
timeevolution.dmaster_nh!
```

```@docs
timeevolution.dmaster_liouville!
```

```@docs
timeevolution.dmaster_h_dynamic!
```

```@docs
timeevolution.dmaster_nh_dynamic!
```

```@docs
timeevolution.@skiptimechecks
```

### [Monte Carlo wave function](@id API: Monte Carlo wave function)

```@docs
timeevolution.mcwf
```

```@docs
timeevolution.mcwf_h
```

```@docs
timeevolution.mcwf_nh
```

```@docs
timeevolution.mcwf_dynamic
```

```@docs
timeevolution.mcwf_nh_dynamic
```

```@docs
diagonaljumps
```

### [Bloch-Redfield master](@id API: Bloch-Redfield)

```@docs
timeevolution.bloch_redfield_tensor
```

```@docs
timeevolution.master_bloch_redfield
```

## [Spectral analysis](@id API: Spectral analysis)

```@docs
eigenstates
```

```@docs
eigenenergies
```

```@docs
simdiag
```


## [Steady-states](@id API: Steady-states)

```@docs
steadystate.master
```

```@docs
steadystate.eigenvector
```

```@docs
steadystate.liouvillianspectrum
```

```@docs
steadystate.iterative
```

```@docs
steadystate.iterative!
```


## [Time correlations](@id API: Time correlations)

```@docs
timecorrelations.correlation
```

```@docs
timecorrelations.spectrum
```

```@docs
timecorrelations.correlation2spectrum
```

```@docs
timecorrelations.correlation_dynamic
```

## [Semi-classical](@id API: Semi-classical systems)

```@docs
semiclassical.State
```

```@docs
semiclassical.schroedinger_dynamic
```

```@docs
semiclassical.dschroedinger_dynamic!
```

```@docs
semiclassical.master_dynamic
```

```@docs
semiclassical.dmaster_h_dynamic!
```

```@docs
semiclassical.mcwf_dynamic
```

## [Stochastics](@id API: Stochastics)

```@docs
stochastic.schroedinger
```

```@docs
stochastic.schroedinger_dynamic
```

```@docs
stochastic.homodyne_carmichael
```

```@docs
stochastic.master
```

```@docs
stochastic.master_dynamic
```

```@docs
stochastic.schroedinger_semiclassical
```

```@docs
stochastic.master_semiclassical
```

## [State definitions](@id API: State definitions)

```@docs
randstate
```

```@docs
thermalstate
```

```@docs
coherentthermalstate
```

```@docs
phase_average
```

```@docs
passive_state
```

```@docs
randstate_haar
```

## [Pauli](@id API: Pauli)

```@docs
PauliTransferMatrix
```

```@docs
DensePauliTransferMatrix
```

```@docs
ChiMatrix
```

```@docs
DenseChiMatrix
```

## [Printing](@id API: Printing)

```@docs
QuantumOpticsBase.set_printing
```

## [LazyTensor functions](@id API: LazyTensor functions)

```@docs
lazytensor_enable_cache
```

```@docs
lazytensor_disable_cache
```

```@docs
lazytensor_cachesize
```

```@docs
lazytensor_clear_cache
```
