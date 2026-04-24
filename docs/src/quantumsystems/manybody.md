# Many-body

```@example
using QuantumOptics # hide
b = NLevelBasis(2)
H = diagonaloperator(b, [0, 1]);

b_mb = ManyBodyBasis(b, bosonstates(b, [0, 1, 2, 3]))

H_mb = manybodyoperator(b_mb, H)
j_mb = transition(b_mb, 1, 2) # Move particles from second to first level
a1_mb = destroy(b_mb, 1) # Destroy particles in first level
at2_mb = create(b_mb, 2) # Create particles in second level
nothing # hide
```

Describing systems consisting of many identical particles in a tensor product space created out of single particle Hilbert spaces leads to the problem that not all states in this space correspond to real physical states. In this picture one would have to restrict the Hilbert space to a subspace that is invariant under permutation of particles. However, it is also possible to find a valid description that doesn't first introduce redundant states that later on have to be eliminated. Instead of asking the question in which state a specific particle is, we count how many particles are in this specific state. The specific choice of this states, which is basically a choice of basis, is physically not relevant but nevertheless has to be done. The general idea is to choose a convenient basis ``\{\left|u_i\right\rangle\}_i`` of the single particle Hilbert space and create the N-particle Hilbert space from states that count how many particles are in each of these states - which will in the following be denoted as ``\left|\{n\}\right\rangle``. Of course the sum of these occupation numbers is identical to the number of particles in this state. For fermionic particles an additional restriction is that there can't be more than one particle in one state. There is no reason to restrict ourself to a certain particle number and therefore a single many-body basis can have basis states corresponding to different amount of particles.

This concept is captured in the [`ManyBodyBasis`](@ref) type:

```julia
struct ManyBodyBasis <: Basis
    shape::Vector{Int}
    onebodybasis::Basis
    occupations::Vector{Vector{Int}}
end
```

Of course we still need a way to relate operators defined in the single particle description to equivalent many-body operators. For an arbitrary additive single particle operator ``\sum_n x_n`` this connection is given by

```math
    X = \sum_{ij} a_i^\dagger a_j
                    \left\langle u_i \right|
                    x
                    \left| u_j \right\rangle
```

For two particle interactions given by the additive two particle operator ``\sum_{m \neq n} v_{mn}`` the corresponding many-body operator is given by:

```math
    V = \sum_{ijkl} a_i^\dagger a_j^\dagger a_k a_l
            \left\langle u_i \right| \left\langle u_j \right|
            v
            \left| u_k \right\rangle \left| u_l \right\rangle
```

The [`manybodyoperator`](@ref) function uses these relations to automatically generate many-body operators from their corresponding one-body operators.


## [States](@id manybody: States)

Since it is typically more natural to label a basis state of a many-body basis by its occupation number, e.g. ``|2,0,1,0\rangle``, instead of calling it the i-th basis state, the [`basisstate`](@ref) function is overloaded to provide exactly this functionality.

* [`basisstate`](@ref)


## [Operators](@id manybody: Operators)

Note that when using the [`create`](@ref) and [`destroy`](@ref) operators for a fixed particle number will most likely fail since every entry of them is zero. Use the [`number`](@ref) and [`transition`](@ref) operators instead.

* [`number`](@ref)
* [`create`](@ref)
* [`destroy`](@ref)
* [`transition`](@ref)

The [`manybodyoperator`](@ref) function allows to automatically generate the many-body operator from the associated one-body operator.

* [`manybodyoperator`](@ref)


## [Examples](@id manybody: Examples)

* [Four level system in many-body formalism](@ref)
* [N-Particles in a double-well potential`](@ref)
