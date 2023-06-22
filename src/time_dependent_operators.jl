# Convert storage of heterogeneous stuff to tuples for maximal compilation
# and to avoid runtime dispatch.
_tuplify(o::TimeDependentSum) = TimeDependentSum(Tuple, o)
_tuplify(o::LazySum) = LazySum(eltype(o.factors), o.factors, (o.operators...,))
_tuplify(o::AbstractOperator) = o

"""
    schroedinger_dynamic_function(H::AbstractTimeDependentOperator)

Creates a function `f(t, state) -> H(t)`. The `state` argument is ignored.

This is the function expected by [`timeevolution.schroedinger_dynamic()`](@ref).
"""
function schroedinger_dynamic_function(H::AbstractTimeDependentOperator)
    _getfunc(op) = (@inline _tdop_schroedinger_wrapper(t, _) = (op)(t))
    Htup = _tuplify(H)
    return _getfunc(Htup)
end

"""
    master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)

Returns a function `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is ignored.

This is the function expected by [`timeevolution.master_h_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
function master_h_dynamic_function(H::AbstractTimeDependentOperator, Js)
    Htup = _tuplify(H)
    Js_tup = ((_tuplify(J) for J in Js)...,)

    # TODO: We can do better than this for TimeDependentSum by only executing
    #       coefficient functions once.
    Jdags_tup = dagger.(Js_tup)
    function _getfunc(Hop, Jops, Jdops)
        return (@inline _tdop_master_wrapper_1(t, _) = ((Hop)(t), set_time!.(Jops, t), set_time!.(Jdops, t)))
    end
    return _getfunc(Htup, Js_tup, Jdags_tup)
end

"""
    master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)

Returns a function `f(t, state) -> Hnh(t), Hnh(t)', Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.master_nh_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the
(time independent) jump operators.
"""
function master_nh_dynamic_function(Hnh::AbstractTimeDependentOperator, Js)
    Hnhtup = _tuplify(Hnh)
    Js_tup = ((_tuplify(J) for J in Js)...,)

    # TODO: We can do better than this for TimeDependentSum by only executing
    #       coefficient functions once.
    Jdags_tup = dagger.(Js_tup)
    Htdagup = dagger(Hnhtup)

    function _getfunc(Hop, Hdop, Jops, Jdops)
        return (@inline _tdop_master_wrapper_2(t, _) = ((Hop)(t), (Hdop)(t), set_time!.(Jops, t), set_time!.(Jdops, t)))
    end
    return _getfunc(Hnhtup, Htdagup, Js_tup, Jdags_tup)
end

"""
    mcfw_dynamic_function(H, Js)

Returns a function `f(t, state) -> H(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `H` is represents the Hamiltonian and `Js` are the (time independent) jump
operators.
"""
mcfw_dynamic_function(H, Js) = master_h_dynamic_function(H, Js)

"""
    mcfw_nh_dynamic_function(Hnh, Js)

Returns a function `f(t, state) -> Hnh(t), Js, dagger.(Js)`.
The `state` argument is currently ignored.

This is the function expected by [`timeevolution.mcwf_dynamic()`](@ref),
where `Hnh` is represents the non-Hermitian Hamiltonian and `Js` are the (time
independent) jump operators.
"""
mcfw_nh_dynamic_function(Hnh, Js) = master_h_dynamic_function(Hnh, Js)
