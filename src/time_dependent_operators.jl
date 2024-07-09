# Convert storage of heterogeneous stuff to tuples for maximal compilation
# and to avoid runtime dispatch.
function _tuplify(o::TimeDependentSum)
    if isconcretetype(eltype(o.coefficients)) && isconcretetype(eltype(o.static_op.operators))
        # No need to tuplify is types are concrete.
        # We will save on compile time this way.
        return o
    end
    return TimeDependentSum(Tuple, o)
end
function _tuplify(o::LazySum)
    if isconcretetype(eltype(o.factors)) && isconcretetype(eltype(o.operators))
        return o
    end
    return LazySum(eltype(o.factors), o.factors, (o.operators...,))
end
_tuplify(o::AbstractOperator) = o

"""
    schroedinger_dynamic_function(H::AbstractTimeDependentOperator)

Creates a function of the form `f(t, state) -> H(t)`. The `state` argument is ignored.

This is the function expected by [`timeevolution.schroedinger_dynamic()`](@ref).
"""
function schroedinger_dynamic_function(H::AbstractTimeDependentOperator)
    _getfunc(op) = (@inline _tdop_schroedinger_wrapper(t, _) = set_time!(op, t))
    Htup = _tuplify(H)
    return _getfunc(Htup)
end

_tdopdagger(o) = dagger(o)
function _tdopdagger(o::TimeDependentSum)
    # This is a kind-of-hacky, more efficient TimeDependentSum dagger operation
    # that requires that the original operator sticks around and is always
    # updated first (though this is checked).
    # Copies and conjugates the coefficients from the original op.
    # TODO: Make an Adjoint wrapper for TimeDependentSum instead?
    o_ls = QuantumOpticsBase.static_operator(o)
    facs = o_ls.factors
    c1 = (t)->(@assert current_time(o) == t; conj(facs[1]))
    crest = (((_)->conj(facs[i])) for i in 2:length(facs))
    odag = TimeDependentSum((c1, crest...), dagger(o_ls), current_time(o))
    return odag
end

"""
    master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)

Returns a function of the form `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is ignored.

This is the function expected by [`timeevolution.master_h_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
function master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)
    Htup = _tuplify(H)
    Js_tup = ((_tuplify(J) for J in Js)...,)
    Jdags_tup = _tdopdagger.(Js_tup)

    return let Hop = Htup, Jops = Js_tup, Jdops = Jdags_tup
        @inline function _tdop_master_wrapper_1(t, _)
            f = (o -> set_time!(o, t))
            foreach(f, Jops)
            foreach(f, Jdops)
            set_time!(Hop, t)
            return Hop, Jops, Jdops
        end
    end
end

"""
    master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)

Returns a function of the form `f(t, state) -> Hnh(t), Hnh(t)', Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.master_nh_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the
(time independent) jump operators.
"""
function master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)
    Hnhtup = _tuplify(Hnh)
    Js_tup = ((_tuplify(J) for J in Js)...,)

    Jdags_tup = _tdopdagger.(Js_tup)
    Htdagup = _tdopdagger(Hnhtup)

    return let Hop = Htup, Hdop = Htdagup, Jops = Js_tup, Jdops = Jdags_tup
        @inline function _tdop_master_wrapper_2(t, _)
            f = (o -> set_time!(o, t))
            foreach(f, Jops)
            foreach(f, Jdops)
            set_time!(Hop, t)
            set_time!(Hdop, t)
            return Hop, Hdop, Jops, Jdops
        end
    end
    return _getfunc(Hnhtup, Htdagup, Js_tup, Jdags_tup)
end

"""
    mcfw_dynamic_function(H, Js)

Returns a function of the form `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
mcfw_dynamic_function(H, Js) = master_h_dynamic_function(H, Js)

"""
    mcfw_nh_dynamic_function(Hnh, Js)

Returns a function of the form `f(t, state) -> Hnh(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the (time
independent) jump operators.
"""
mcfw_nh_dynamic_function(Hnh, Js) = master_h_dynamic_function(Hnh, Js)
