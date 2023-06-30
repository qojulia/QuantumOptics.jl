using Test
using QuantumOptics

@testset "time-dependent operators" begin

b = FockBasis(7)

a = destroy(b)

H0 = number(b)
Hd = (a + a')
H = TimeDependentSum(1.0=>H0, cos=>Hd)

ts = [0.0, 0.4]
ts_half = 0.5 * ts

_h(t, H0, Hd) = H0 + cos(t)*Hd
_getf = (H0, Hd) -> (t,_) -> _h(t, H0, Hd)
fman = _getf(H0, Hd)

psi0 = basisstate(b, 1)
ts_out, psis = timeevolution.schroedinger_dynamic(ts, psi0, H)
# check this is not trivial
@test !(psis[1].data ≈ psis[end].data)

ts_out2, psis2 = timeevolution.schroedinger_dynamic(ts, psi0, fman)
@test psis[end].data ≈ psis2[end].data

_getf = (H0, Hd, a) -> (t,_) -> (_h(t, H0, Hd), (), ())
fman = _getf(H0, Hd, a)

ts_out, rhos = timeevolution.master_dynamic(ts, psi0, H, [])
ts_out2, rhos2 = timeevolution.master_dynamic(ts, psi0, fman)
@test rhos[end].data ≈ rhos2[end].data

Js = [TimeDependentSum(cos=>a), 0.01 * a', 0.01 * LazySum(a' * a)]
Jdags = dagger.(Js)

_js(t, a) = (cos(t)*a, 0.01*a', 0.01*a'*a)

_getf = (H0, Hd, a) -> (t,_) -> (_h(t, H0, Hd), _js(t, a), dagger.(_js(t, a)))
fman = _getf(H0, Hd, a)

ts_out, psis = timeevolution.mcwf_dynamic(ts, psi0, H, Js; seed=0)
ts_out2, psis2 = timeevolution.mcwf_dynamic(ts, psi0, fman; seed=0)
@test psis[end].data ≈ psis2[end].data

ts_out, rhos = timeevolution.master_dynamic(ts, psi0, H, Js)
ts_out2, rhos2 = timeevolution.master_dynamic(ts, psi0, fman)
@test rhos[end].data ≈ rhos2[end].data

Hnh = H - 0.5im * sum(J' * J for J in Js)

_getf = (H0, Hd, a) -> (t,_) -> (
    _h(t, H0, Hd) - 0.5im * sum(dagger.(_js(t, a)) .* _js(t, a)),
    _js(t, a),
    dagger.(_js(t, a)))

fman = _getf(H0, Hd, a)

ts_out, psis = timeevolution.mcwf_nh_dynamic(ts, psi0, Hnh, Js; seed=0)
ts_out2, psis2 = timeevolution.mcwf_nh_dynamic(ts, psi0, fman; seed=0)
@test psis[end].data ≈ psis2[end].data

_getf = (H0, Hd, a) -> (t,_) -> (
    _h(t, H0, Hd) - 0.5im * sum(dagger.(_js(t, a)) .* _js(t, a)),
    _h(t, H0, Hd) + 0.5im * sum(dagger.(_js(t, a)) .* _js(t, a)),
    _js(t, a),
    dagger.(_js(t, a)))

fman = _getf(H0, Hd, a)

ts_out, rhos = timeevolution.master_nh_dynamic(ts, psi0, Hnh, Js)
ts_out2, rhos2 = timeevolution.master_nh_dynamic(ts, psi0, fman)
@test rhos[end].data ≈ rhos2[end].data

# for sparse operators, we should not be allocating at each timestep
allocs1 = @allocated timeevolution.schroedinger_dynamic(ts, psi0, H)
allocs2 = @allocated timeevolution.schroedinger_dynamic(ts_half, psi0, H)
@test allocs1 == allocs2

allocs1 = @allocated timeevolution.master_nh_dynamic(ts, psi0, Hnh, Js)
allocs2 = @allocated timeevolution.master_nh_dynamic(ts_half, psi0, Hnh, Js)
@test allocs1 == allocs2

end
