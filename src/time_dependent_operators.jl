# Convert storage of heterogeneous stuff to tuples for maximal compilation
# and to avoid runtime dispatch.
_tuplify(o::TimeDependentSum) = TimeDependentSum(Tuple, o)
_tuplify(o::LazySum) = LazySum(eltype(o.factors), o.factors, (o.operators...,))
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
    function _getfunc(Hop, Jops, Jdops)
        return (@inline _tdop_master_wrapper_1(t, _) = (set_time!(Hop, t), set_time!.(Jops, t), set_time!.(Jdops, t)))
    end
    return _getfunc(Htup, Js_tup, Jdags_tup)
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

    function _getfunc(Hop, Hdop, Jops, Jdops)
        return (@inline _tdop_master_wrapper_2(t, _) = (set_time!(Hop, t), set_time!(Hdop, t), set_time!.(Jops, t), set_time!.(Jdops, t)))
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
