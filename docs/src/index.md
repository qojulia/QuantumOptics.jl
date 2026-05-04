# Introduction

**QuantumOptics.jl** is a numerical framework written in [Julia](http://julialang.org/) that makes it easy to simulate various kinds of quantum systems. It is similar to the [Quantum Optics Toolbox](http://qo.phy.auckland.ac.nz/toolbox/) for MATLAB and its Python successor [QuTiP](http://qutip.org/).

One of the core concepts of **QuantumOptics.jl** is that all quantum objects, i.e. state vectors and operators have knowledge about which Hilbert space they live in. This prevents many common mistakes when working with composite systems and at the same time improves readability. The Hilbert spaces are defined implicitly by specifying appropriate bases like a [`FockBasis`](@ref) or a [`SpinBasis`](@ref). These bases can in turn be combined to describe composite systems like e.g. a particle in a cavity or a multi-spin system. The different kinds of bases that are implemented are introduced in [Quantum systems](@ref).

After choosing a basis, **QuantumOptics.jl** provides many useful functions to create common [States](@ref) and [Operators](@ref) which can be combined in all the expected ways. Consequently, constructing arbitrary Hamiltonians and Liouvillians and specifying initial states is straightforward. These objects can be used to perform a [Time-evolution](@ref).

Although the main focus is on simulating dynamics of (open) quantum systems, there are nevertheless additional features available, for example to calculate a [Steady state](@ref) or [Two-time correlation functions](@ref).

For a quick introduction it is probably best to start reading the [Tutorial](@ref). It shows a typical approach to study a quantum system and provides links to the corresponding topics in the documentation.
