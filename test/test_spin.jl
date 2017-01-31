using Base.Test
using QuantumOptics

distance(a::SparseOperator, b::SparseOperator) = tracedistance(full(a), full(b))

for spinnumber=1//2:1//2:5//2
    spinbasis = SpinBasis(spinnumber)
    I = operators.identityoperator(spinbasis)
    Zero = SparseOperator(spinbasis)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)


    # Test traces
    @test_approx_eq 0. trace(sx)
    @test_approx_eq 0. trace(sy)
    @test_approx_eq 0. trace(sz)


    # Test kommutation relations
    kommutator(x, y) = x*y - y*x

    @test_approx_eq_eps 0. distance(kommutator(sx, sx), Zero) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sx, sy), 2im*sz) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sx, sz), -2im*sy) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, sx), -2im*sz) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, sy), Zero) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, sz), 2im*sx) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, sx), 2im*sy) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, sy), -2im*sx) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, sz), Zero) 1e-12


    # Test creation and anihilation operators
    @test_approx_eq 0. distance(sp, 0.5*(sx + 1im*sy))
    @test_approx_eq 0. distance(sm, 0.5*(sx - 1im*sy))
    @test_approx_eq 0. distance(sx, (sp + sm))
    @test_approx_eq 0. distance(sy, -1im*(sp - sm))


    # Test commutation relations with creation and anihilation operators
    @test_approx_eq_eps 0. distance(kommutator(sp, sm), sz) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, sp), 2*sp) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, sm), -2*sm) 1e-12


    # Test v x (v x u) relation: [sa, [sa, sb]] = 4*(1-delta_{ab})*sb
    @test_approx_eq_eps 0. distance(kommutator(sx, kommutator(sx, sx)), Zero) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sx, kommutator(sx, sy)), 4*sy) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sx, kommutator(sx, sz)), 4*sz) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, kommutator(sy, sx)), 4*sx) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, kommutator(sy, sy)), Zero) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sy, kommutator(sy, sz)), 4*sz) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, kommutator(sz, sx)), 4*sx) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, kommutator(sz, sy)), 4*sy) 1e-12
    @test_approx_eq_eps 0. distance(kommutator(sz, kommutator(sz, sz)), Zero) 1e-12


    # Test spinup and spindown states
    @test_approx_eq_eps 1. norm(spinup(spinbasis)) 1e-11
    @test_approx_eq_eps 1. norm(spindown(spinbasis)) 1e-11
    @test_approx_eq_eps 0. norm(sp*spinup(spinbasis)) 1e-11
    @test_approx_eq_eps 0. norm(sm*spindown(spinbasis)) 1e-11
end


# Test special relations for spin 1/2

spinbasis = SpinBasis(1//2)
I = identityoperator(spinbasis)
Zero = SparseOperator(spinbasis)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)


# Test antikommutator
antikommutator(x, y) = x*y + y*x

@test_approx_eq 0. distance(antikommutator(sx, sx), 2*I)
@test_approx_eq 0. distance(antikommutator(sx, sy), Zero)
@test_approx_eq 0. distance(antikommutator(sx, sz), Zero)
@test_approx_eq 0. distance(antikommutator(sy, sx), Zero)
@test_approx_eq 0. distance(antikommutator(sy, sy), 2*I)
@test_approx_eq 0. distance(antikommutator(sy, sz), Zero)
@test_approx_eq 0. distance(antikommutator(sz, sx), Zero)
@test_approx_eq 0. distance(antikommutator(sz, sy), Zero)
@test_approx_eq 0. distance(antikommutator(sz, sz), 2*I)


# Test if involutory for spin 1/2
@test_approx_eq 0. distance(sx*sx, I)
@test_approx_eq 0. distance(sy*sy, I)
@test_approx_eq 0. distance(sz*sz, I)
@test_approx_eq 0. distance(-1im*sx*sy*sz, I)


# Test consistency of spin up and down with sigmap and sigmam
@test_approx_eq_eps 0. norm(sm*spinup(spinbasis) - spindown(spinbasis)) 1e-11
@test_approx_eq_eps 0. norm(sp*spindown(spinbasis) - spinup(spinbasis)) 1e-11
